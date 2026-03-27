import json
import logging
import os
from typing import Any, AsyncIterator

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from litellm import acompletion

try:
    from logger import logger_api  # type: ignore
except Exception:
    def logger_api(request: dict[str, Any], model_name: str, response: Any) -> None:
        logging.warning("logger_api not found in logger module, fallback logger used.")
        logging.info(
            "request_model=%s request=%s response=%s",
            model_name,
            request,
            response,
        )


logging.basicConfig(level=logging.INFO)
app = FastAPI(title="LiteLLM Forwarder v2")

# 和 main.py 保持一致，方便直接复用环境变量。
UPSTREAM_BASE_URL = os.getenv("UPSTREAM_BASE_URL", "https://api.gptplus5.com/v1").rstrip("/")
UPSTREAM_API_KEY = os.getenv("UPSTREAM_API_KEY", "")
DEFAULT_TIMEOUT_SECONDS = float(os.getenv("UPSTREAM_TIMEOUT_SECONDS", "120"))


def _extract_upstream_api_key(incoming_headers: dict[str, str]) -> str:
    auth = incoming_headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth.split(" ", 1)[1].strip()
    return UPSTREAM_API_KEY


def _build_upstream_headers(incoming_headers: dict[str, str]) -> dict[str, str]:
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    auth = incoming_headers.get("authorization")
    if auth:
        headers["Authorization"] = auth
    elif UPSTREAM_API_KEY:
        headers["Authorization"] = f"Bearer {UPSTREAM_API_KEY}"
    return headers


async def _parse_json_payload(request: Request) -> dict[str, Any]:
    try:
        payload = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JSON body: {exc}") from exc
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Request body must be a JSON object.")
    return payload


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    payload = await _parse_json_payload(request)
    model_name = str(payload.get("model", "unknown"))
    incoming_headers = dict(request.headers)
    upstream_api_key = _extract_upstream_api_key(incoming_headers)
    stream = bool(payload.get("stream", False))

    if stream:
        async def stream_generator() -> AsyncIterator[bytes]:
            raw_chunks: list[str] = []
            try:
                stream_resp = await acompletion(
                    **payload,
                    api_base=UPSTREAM_BASE_URL,
                    api_key=upstream_api_key,
                    timeout=DEFAULT_TIMEOUT_SECONDS,
                )
                async for chunk in stream_resp:
                    if hasattr(chunk, "model_dump_json"):
                        chunk_text = chunk.model_dump_json()
                    elif hasattr(chunk, "json"):
                        chunk_text = chunk.json()
                    elif isinstance(chunk, bytes):
                        chunk_text = chunk.decode("utf-8", errors="ignore")
                    elif isinstance(chunk, str):
                        chunk_text = chunk
                    else:
                        chunk_text = json.dumps(chunk, ensure_ascii=False, default=str)
                    raw_chunks.append(chunk_text)
                    yield f"data: {chunk_text}\n\n".encode("utf-8")
                yield b"data: [DONE]\n\n"
            except Exception as exc:
                status_code = int(getattr(exc, "status_code", 502))
                raise HTTPException(status_code=status_code, detail=f"LiteLLM stream request failed: {exc}") from exc
            finally:
                try:
                    logger_api(payload, model_name, "".join(raw_chunks))
                except Exception as log_exc:
                    logging.exception("logger_api failed in stream mode: %s", log_exc)

        return StreamingResponse(stream_generator(), media_type="text/event-stream")

    try:
        result = await acompletion(
            **payload,
            api_base=UPSTREAM_BASE_URL,
            api_key=upstream_api_key,
            timeout=DEFAULT_TIMEOUT_SECONDS,
        )
    except Exception as exc:
        status_code = int(getattr(exc, "status_code", 502))
        raise HTTPException(status_code=status_code, detail=f"LiteLLM request failed: {exc}") from exc

    if hasattr(result, "model_dump"):
        body: Any = result.model_dump()
    elif isinstance(result, dict):
        body = result
    else:
        body = {"raw_text": str(result)}

    try:
        logger_api(payload, model_name, body)
    except Exception as log_exc:
        logging.exception("logger_api failed: %s", log_exc)

    if isinstance(body, dict):
        return JSONResponse(status_code=200, content=body)
    return JSONResponse(status_code=200, content={"data": body})


@app.api_route("/v1/{full_path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
async def passthrough_v1(full_path: str, request: Request):
    # chat/completions 走 LiteLLM，其他接口直接透传到上游，保持兼容性。
    if full_path == "chat/completions":
        return await chat_completions(request)

    upstream_url = f"{UPSTREAM_BASE_URL}/v1/{full_path}"
    headers = _build_upstream_headers(dict(request.headers))
    method = request.method.upper()
    timeout = httpx.Timeout(DEFAULT_TIMEOUT_SECONDS)
    raw_body = await request.body()

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            upstream_resp = await client.request(
                method=method,
                url=upstream_url,
                headers=headers,
                content=raw_body if raw_body else None,
                params=request.query_params,
            )
    except httpx.RequestError as exc:
        raise HTTPException(status_code=502, detail=f"Upstream request failed: {exc}") from exc

    model_name = "passthrough"
    req_for_log: dict[str, Any] = {
        "path": f"/v1/{full_path}",
        "method": method,
    }
    if raw_body:
        try:
            req_for_log["body"] = json.loads(raw_body.decode("utf-8"))
            model_name = str(req_for_log["body"].get("model", "passthrough"))
        except Exception:
            req_for_log["body"] = raw_body.decode("utf-8", errors="ignore")

    resp_for_log: Any
    content_type = upstream_resp.headers.get("content-type", "")
    if "application/json" in content_type:
        try:
            resp_for_log = upstream_resp.json()
        except json.JSONDecodeError:
            resp_for_log = {"raw_text": upstream_resp.text}
    else:
        resp_for_log = {"raw_text": upstream_resp.text}

    try:
        logger_api(req_for_log, model_name, resp_for_log)
    except Exception as log_exc:
        logging.exception("logger_api failed in passthrough endpoint: %s", log_exc)

    filtered_headers = {
        k: v
        for k, v in upstream_resp.headers.items()
        if k.lower() not in {"content-length", "transfer-encoding", "connection"}
    }
    return Response(
        content=upstream_resp.content,
        status_code=upstream_resp.status_code,
        headers=filtered_headers,
        media_type=upstream_resp.headers.get("content-type"),
    )
