import json
import logging
import os
from typing import Any, AsyncIterator

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

try:
    # 如果你已有独立 logger 模块，保持这个导入即可。
    from logger import logger_api  # type: ignore
except Exception:
    def logger_api(request: dict[str, Any], model_name: str, response: Any) -> None:
        logging.warning("logger_api not found in logger module, fallback logger used.")
        logging.info(
            "request_model=%s request=%s response=%s",
            model_name,
            request,
            response
        )


logging.basicConfig(level=logging.INFO)
app = FastAPI(title="OpenAI-Compatible Forwarder with Logging")

UPSTREAM_BASE_URL = os.getenv("UPSTREAM_BASE_URL", "https://api.gptplus5.com/v1").rstrip("/")
UPSTREAM_API_KEY = os.getenv("UPSTREAM_API_KEY", "")
DEFAULT_TIMEOUT_SECONDS = float(os.getenv("UPSTREAM_TIMEOUT_SECONDS", "120"))


def _build_upstream_headers(incoming_headers: dict[str, str]) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    auth = incoming_headers.get("authorization")
    if auth:
        headers["Authorization"] = auth
    elif UPSTREAM_API_KEY:
        headers["Authorization"] = f"Bearer {UPSTREAM_API_KEY}"
    return headers


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/chat/completions")
async def forward_chat_completions(request: Request):
    if not UPSTREAM_BASE_URL:
        raise HTTPException(
            status_code=500,
            detail="UPSTREAM_BASE_URL is not configured.",
        )

    try:
        payload = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JSON body: {exc}") from exc

    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Request body must be a JSON object.")

    model_name = str(payload.get("model", "unknown"))
    upstream_url = f"{UPSTREAM_BASE_URL}/v1/chat/completions"
    headers = _build_upstream_headers(dict(request.headers))
    timeout = httpx.Timeout(DEFAULT_TIMEOUT_SECONDS)
    stream = bool(payload.get("stream", False))

    if stream:
        client = httpx.AsyncClient(timeout=timeout)
        try:
            upstream_req = client.build_request(
                "POST",
                upstream_url,
                headers=headers,
                json=payload,
            )
            upstream_resp = await client.send(upstream_req, stream=True)
        except httpx.RequestError as exc:
            await client.aclose()
            raise HTTPException(status_code=502, detail=f"Upstream request failed: {exc}") from exc

        async def stream_generator() -> AsyncIterator[bytes]:
            chunks: list[bytes] = []
            try:
                async for chunk in upstream_resp.aiter_bytes():
                    if chunk:
                        chunks.append(chunk)
                        yield chunk
            finally:
                try:
                    raw_response = b"".join(chunks).decode("utf-8", errors="ignore")
                    # 流式场景通常是 SSE 文本，直接记录原始内容更稳妥。
                    logger_api(payload, model_name, raw_response)
                except Exception as log_exc:
                    logging.exception("logger_api failed in stream mode: %s", log_exc)
                await upstream_resp.aclose()
                await client.aclose()

        response_headers = {}
        if "content-type" in upstream_resp.headers:
            response_headers["content-type"] = upstream_resp.headers["content-type"]

        return StreamingResponse(
            stream_generator(),
            status_code=upstream_resp.status_code,
            headers=response_headers,
        )

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            upstream_resp = await client.post(
                upstream_url,
                headers=headers,
                json=payload,
            )
    except httpx.RequestError as exc:
        raise HTTPException(status_code=502, detail=f"Upstream request failed: {exc}") from exc

    content_type = upstream_resp.headers.get("content-type", "")
    if "application/json" in content_type:
        try:
            response_body: Any = upstream_resp.json()
        except json.JSONDecodeError:
            response_body = {"raw_text": upstream_resp.text}
    else:
        response_body = {"raw_text": upstream_resp.text}

    try:
        logger_api(payload, model_name, response_body)
    except Exception as log_exc:
        logging.exception("logger_api failed: %s", log_exc)

    if isinstance(response_body, dict):
        return JSONResponse(status_code=upstream_resp.status_code, content=response_body)
    return JSONResponse(status_code=upstream_resp.status_code, content={"data": response_body})


@app.get("/v1/models")
async def forward_models(request: Request):
    if not UPSTREAM_BASE_URL:
        raise HTTPException(
            status_code=500,
            detail="UPSTREAM_BASE_URL is not configured.",
        )

    upstream_url = f"{UPSTREAM_BASE_URL}/v1/models"
    headers = _build_upstream_headers(dict(request.headers))
    timeout = httpx.Timeout(DEFAULT_TIMEOUT_SECONDS)

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            upstream_resp = await client.get(upstream_url, headers=headers)
    except httpx.RequestError as exc:
        raise HTTPException(status_code=502, detail=f"Upstream request failed: {exc}") from exc

    content_type = upstream_resp.headers.get("content-type", "")
    if "application/json" in content_type:
        try:
            response_body: Any = upstream_resp.json()
        except json.JSONDecodeError:
            response_body = {"raw_text": upstream_resp.text}
    else:
        response_body = {"raw_text": upstream_resp.text}

    # models 列表接口没有请求体，这里传入空对象用于统一日志结构。
    try:
        logger_api({}, "models", response_body)
    except Exception as log_exc:
        logging.exception("logger_api failed in models endpoint: %s", log_exc)

    if isinstance(response_body, dict):
        return JSONResponse(status_code=upstream_resp.status_code, content=response_body)
    return JSONResponse(status_code=upstream_resp.status_code, content={"data": response_body})

