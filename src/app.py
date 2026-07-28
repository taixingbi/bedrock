import json
import os
import time
import uuid
from collections.abc import Iterator
from typing import Any

import boto3
from botocore.exceptions import ClientError
from fastapi import FastAPI, Header, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

bedrock = boto3.client("bedrock-runtime")

DEFAULT_MODEL_ID = os.environ.get("MODEL_ID", "amazon.nova-lite-v1:0")
# Back-compat alias used in a few call sites / docs.
MODEL_ID = DEFAULT_MODEL_ID
API_KEY = os.environ.get("API_KEY", "")

# Friendly request names → Bedrock model ID / imported-model ARN.
_BUILTIN_MODEL_ALIASES: dict[str, str] = {
    "claude-sonnet": "anthropic.claude-sonnet-5",
    "claude-sonnet-5": "anthropic.claude-sonnet-5",
    "anthropic.claude-sonnet-5": "anthropic.claude-sonnet-5",
    "us.anthropic.claude-sonnet-5": "us.anthropic.claude-sonnet-5",
    "global.anthropic.claude-sonnet-5": "global.anthropic.claude-sonnet-5",
    "amazon.nova-lite-v1:0": "amazon.nova-lite-v1:0",
    "nova-lite": "amazon.nova-lite-v1:0",
    "amazon.nova-pro-v1:0": "amazon.nova-pro-v1:0",
    "nova-pro": "amazon.nova-pro-v1:0",
    "us.amazon.nova-pro-v1:0": "us.amazon.nova-pro-v1:0",
}


def _is_imported_model(model_id: str) -> bool:
    return ":imported-model/" in model_id


def _load_model_map() -> dict[str, str]:
    mapping = dict(_BUILTIN_MODEL_ALIASES)
    mapping[DEFAULT_MODEL_ID] = DEFAULT_MODEL_ID
    if _is_imported_model(DEFAULT_MODEL_ID):
        mapping.setdefault("Qwen/Qwen2.5-7B-Instruct", DEFAULT_MODEL_ID)
        mapping.setdefault("qwen2.5-7b-instruct", DEFAULT_MODEL_ID)
        mapping.setdefault("qwen", DEFAULT_MODEL_ID)

    raw = os.environ.get("MODEL_MAP", "").strip()
    if raw:
        try:
            extra = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"MODEL_MAP must be valid JSON: {exc}") from exc
        if not isinstance(extra, dict):
            raise RuntimeError("MODEL_MAP must be a JSON object of alias → bedrock id")
        for key, value in extra.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise RuntimeError("MODEL_MAP keys and values must be strings")
            mapping[key] = value
    return mapping


MODEL_MAP = _load_model_map()

app = FastAPI(title="mvp-bedrock")


def _resolve_model(request_model: Any) -> tuple[str, str]:
    """Return (response_model_name, bedrock_model_id)."""
    if request_model is None or request_model == "":
        return DEFAULT_MODEL_ID, DEFAULT_MODEL_ID
    if not isinstance(request_model, str):
        raise ValueError("model must be a string")

    name = request_model.strip()
    if name in MODEL_MAP:
        return name, MODEL_MAP[name]

    # Allow raw Bedrock IDs / ARNs / inference profiles not listed in the map.
    if (
        name.startswith("arn:aws:bedrock:")
        or name.startswith("anthropic.")
        or name.startswith("amazon.")
        or name.startswith("us.")
        or name.startswith("eu.")
        or name.startswith("au.")
        or name.startswith("global.")
    ):
        return name, name

    known = ", ".join(sorted(MODEL_MAP))
    raise ValueError(f"unknown model '{name}'; known: {known}")


def _authorized(x_api_key: str | None, authorization: str | None) -> bool:
    if not API_KEY:
        return False
    if x_api_key == API_KEY:
        return True
    if authorization and authorization.lower().startswith("bearer "):
        return authorization[7:].strip() == API_KEY
    return False


def _unauthorized() -> JSONResponse:
    return JSONResponse(status_code=401, content={"error": "unauthorized"})


def _extract_converse_text(converse_response: dict[str, Any]) -> str:
    parts: list[str] = []
    message = converse_response.get("output", {}).get("message", {})
    for block in message.get("content", []):
        text = block.get("text")
        if text:
            parts.append(text)
    return "".join(parts)


def _extract_invoke_text(invoke_body: dict[str, Any]) -> str:
    choices = invoke_body.get("choices")
    if isinstance(choices, list) and choices:
        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if isinstance(block, dict) and isinstance(block.get("text"), str):
                    parts.append(block["text"])
                elif isinstance(block, str):
                    parts.append(block)
            return "".join(parts)

    generation = invoke_body.get("generation")
    if isinstance(generation, str):
        return generation

    outputs = invoke_body.get("outputs")
    if isinstance(outputs, list) and outputs:
        text = outputs[0].get("text")
        if isinstance(text, str):
            return text

    return ""


def _parse_sampling(payload: dict[str, Any]) -> tuple[int, float | None, float | None]:
    max_tokens = payload.get("max_tokens", 512)
    if not isinstance(max_tokens, int) or max_tokens < 1 or max_tokens > 4096:
        raise ValueError("max_tokens must be an integer between 1 and 4096")

    temperature = payload.get("temperature")
    if temperature is not None and (
        not isinstance(temperature, (int, float)) or temperature < 0 or temperature > 2
    ):
        raise ValueError("temperature must be a number between 0 and 2")

    top_p = payload.get("top_p")
    if top_p is not None and (not isinstance(top_p, (int, float)) or top_p <= 0 or top_p > 1):
        raise ValueError("top_p must be a number between 0 and 1")

    return (
        max_tokens,
        float(temperature) if temperature is not None else None,
        float(top_p) if top_p is not None else None,
    )


def _normalize_messages(payload: dict[str, Any]) -> list[dict[str, str]]:
    messages = payload.get("messages")
    if isinstance(messages, list) and messages:
        normalized: list[dict[str, str]] = []
        for item in messages:
            if not isinstance(item, dict):
                raise ValueError("each message must be an object")
            role = item.get("role")
            content = item.get("content")
            if role not in ("system", "user", "assistant"):
                raise ValueError("message.role must be system, user, or assistant")
            if not isinstance(content, str):
                raise ValueError("message.content must be a string")
            normalized.append({"role": role, "content": content})
        if not any(m["role"] == "user" for m in normalized):
            raise ValueError("at least one user message is required")
        return normalized

    prompt = payload.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("messages or prompt is required")

    normalized: list[dict[str, str]] = []
    system = payload.get("system")
    if isinstance(system, str) and system.strip():
        normalized.append({"role": "system", "content": system})
    normalized.append({"role": "user", "content": prompt})
    return normalized


def _split_system(messages: list[dict[str, str]]) -> tuple[str | None, list[dict[str, str]]]:
    system_parts = [m["content"] for m in messages if m["role"] == "system"]
    rest = [m for m in messages if m["role"] != "system"]
    system = "\n\n".join(system_parts) if system_parts else None
    return system, rest


def _invoke_body(
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float | None,
    top_p: float | None,
    *,
    stream: bool,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": stream,
    }
    if temperature is not None:
        body["temperature"] = temperature
    if top_p is not None:
        body["top_p"] = top_p
    return body


def _infer_converse(
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float | None,
    top_p: float | None,
    bedrock_model_id: str,
) -> dict[str, Any]:
    system, rest = _split_system(messages)
    converse_messages = [
        {"role": m["role"], "content": [{"text": m["content"]}]} for m in rest
    ]
    inference_config: dict[str, Any] = {"maxTokens": max_tokens}
    if temperature is not None:
        inference_config["temperature"] = temperature
    if top_p is not None:
        inference_config["topP"] = top_p

    converse_args: dict[str, Any] = {
        "modelId": bedrock_model_id,
        "messages": converse_messages,
        "inferenceConfig": inference_config,
    }
    if system:
        converse_args["system"] = [{"text": system}]

    result = bedrock.converse(**converse_args)
    usage = result.get("usage") or {}
    return {
        "text": _extract_converse_text(result),
        "usage": {
            "prompt_tokens": usage.get("inputTokens", 0),
            "completion_tokens": usage.get("outputTokens", 0),
        },
    }


def _infer_invoke_model(
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float | None,
    top_p: float | None,
    bedrock_model_id: str,
) -> dict[str, Any]:
    raw = bedrock.invoke_model(
        modelId=bedrock_model_id,
        contentType="application/json",
        accept="application/json",
        body=json.dumps(
            _invoke_body(messages, max_tokens, temperature, top_p, stream=False)
        ),
    )
    response_body = json.loads(raw["body"].read())
    usage = response_body.get("usage") or {}
    prompt_tokens = (
        usage.get("prompt_tokens")
        or usage.get("inputTokens")
        or usage.get("input_tokens")
        or 0
    )
    completion_tokens = (
        usage.get("completion_tokens")
        or usage.get("outputTokens")
        or usage.get("output_tokens")
        or 0
    )
    return {
        "text": _extract_invoke_text(response_body),
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        },
    }


def _openai_completion(model: str, text: str, usage: dict[str, int]) -> dict[str, Any]:
    prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
    completion_tokens = int(usage.get("completion_tokens", 0) or 0)
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def _legacy_infer_response(model: str, text: str, usage: dict[str, int]) -> dict[str, Any]:
    return {
        "text": text,
        "model": model,
        "usage": {
            "input_tokens": int(usage.get("prompt_tokens", 0) or 0),
            "output_tokens": int(usage.get("completion_tokens", 0) or 0),
        },
    }


def _sse(data: str) -> str:
    return f"data: {data}\n\n"


def _openai_chunk(
    *,
    completion_id: str,
    created: int,
    model: str,
    delta: dict[str, Any],
    finish_reason: str | None = None,
) -> dict[str, Any]:
    return {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }


def _extract_stream_delta_text(chunk: dict[str, Any]) -> tuple[str, str | None]:
    """Return (text, finish_reason) from an imported-model stream chunk."""
    choices = chunk.get("choices")
    if isinstance(choices, list) and choices:
        choice = choices[0] or {}
        finish_reason = choice.get("finish_reason")
        delta = choice.get("delta") or {}
        if isinstance(delta, dict):
            content = delta.get("content")
            if isinstance(content, str) and content:
                return content, finish_reason if isinstance(finish_reason, str) else None
        message = choice.get("message") or {}
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str) and content:
                return content, finish_reason if isinstance(finish_reason, str) else None
        text = choice.get("text")
        if isinstance(text, str) and text:
            return text, finish_reason if isinstance(finish_reason, str) else None

    for key in ("generation", "completion", "outputText", "text"):
        value = chunk.get(key)
        if isinstance(value, str) and value:
            return value, None

    outputs = chunk.get("outputs")
    if isinstance(outputs, list) and outputs:
        text = outputs[0].get("text")
        if isinstance(text, str) and text:
            return text, None

    return "", None


def _stream_invoke_model(
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float | None,
    top_p: float | None,
    model: str,
    bedrock_model_id: str,
) -> Iterator[str]:
    # Open Bedrock stream before emitting SSE so setup failures become HTTP 502.
    response = bedrock.invoke_model_with_response_stream(
        modelId=bedrock_model_id,
        contentType="application/json",
        accept="application/json",
        body=json.dumps(
            _invoke_body(messages, max_tokens, temperature, top_p, stream=True)
        ),
    )
    event_stream = response.get("body")

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())
    yield _sse(
        json.dumps(
            _openai_chunk(
                completion_id=completion_id,
                created=created,
                model=model,
                delta={"role": "assistant", "content": ""},
            )
        )
    )

    finish_reason: str | None = None
    for event in event_stream:
        chunk_event = event.get("chunk")
        if not chunk_event:
            error_keys = [
                "internalServerException",
                "modelStreamErrorException",
                "validationException",
                "throttlingException",
                "modelTimeoutException",
                "serviceUnavailableException",
            ]
            for key in error_keys:
                if key in event:
                    message = (event[key] or {}).get("message") or key
                    raise RuntimeError(message)
            continue

        payload = json.loads(chunk_event["bytes"])
        text, chunk_finish = _extract_stream_delta_text(payload)
        if chunk_finish:
            finish_reason = chunk_finish
        if text:
            yield _sse(
                json.dumps(
                    _openai_chunk(
                        completion_id=completion_id,
                        created=created,
                        model=model,
                        delta={"content": text},
                    )
                )
            )

    yield _sse(
        json.dumps(
            _openai_chunk(
                completion_id=completion_id,
                created=created,
                model=model,
                delta={},
                finish_reason=finish_reason or "stop",
            )
        )
    )
    yield _sse("[DONE]")


def _stream_converse(
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float | None,
    top_p: float | None,
    model: str,
    bedrock_model_id: str,
) -> Iterator[str]:
    system, rest = _split_system(messages)
    converse_messages = [
        {"role": m["role"], "content": [{"text": m["content"]}]} for m in rest
    ]
    inference_config: dict[str, Any] = {"maxTokens": max_tokens}
    if temperature is not None:
        inference_config["temperature"] = temperature
    if top_p is not None:
        inference_config["topP"] = top_p

    converse_args: dict[str, Any] = {
        "modelId": bedrock_model_id,
        "messages": converse_messages,
        "inferenceConfig": inference_config,
    }
    if system:
        converse_args["system"] = [{"text": system}]

    response = bedrock.converse_stream(**converse_args)
    event_stream = response.get("stream", [])

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())
    yield _sse(
        json.dumps(
            _openai_chunk(
                completion_id=completion_id,
                created=created,
                model=model,
                delta={"role": "assistant", "content": ""},
            )
        )
    )

    finish_reason = "stop"
    for event in event_stream:
        if "contentBlockDelta" in event:
            delta = event["contentBlockDelta"].get("delta") or {}
            text = delta.get("text")
            if isinstance(text, str) and text:
                yield _sse(
                    json.dumps(
                        _openai_chunk(
                            completion_id=completion_id,
                            created=created,
                            model=model,
                            delta={"content": text},
                        )
                    )
                )
        elif "messageStop" in event:
            stop_reason = event["messageStop"].get("stopReason")
            finish_reason = "length" if stop_reason == "max_tokens" else "stop"

    yield _sse(
        json.dumps(
            _openai_chunk(
                completion_id=completion_id,
                created=created,
                model=model,
                delta={},
                finish_reason=finish_reason,
            )
        )
    )
    yield _sse("[DONE]")


def _run_inference(
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float | None,
    top_p: float | None,
    bedrock_model_id: str,
) -> dict[str, Any]:
    if _is_imported_model(bedrock_model_id):
        return _infer_invoke_model(
            messages, max_tokens, temperature, top_p, bedrock_model_id
        )
    return _infer_converse(
        messages, max_tokens, temperature, top_p, bedrock_model_id
    )


def _stream_inference(
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float | None,
    top_p: float | None,
    model: str,
    bedrock_model_id: str,
) -> Iterator[str]:
    if _is_imported_model(bedrock_model_id):
        return _stream_invoke_model(
            messages, max_tokens, temperature, top_p, model, bedrock_model_id
        )
    return _stream_converse(
        messages, max_tokens, temperature, top_p, model, bedrock_model_id
    )


def _bedrock_error(exc: Exception) -> JSONResponse:
    if isinstance(exc, ClientError):
        detail = exc.response.get("Error", {}).get("Message", str(exc))
    else:
        detail = str(exc)
    return JSONResponse(
        status_code=502,
        content={"error": "bedrock request failed", "detail": detail},
    )


@app.options("/{full_path:path}")
async def options(full_path: str) -> Response:  # noqa: ARG001
    return Response(
        status_code=204,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "content-type,x-api-key,authorization",
            "Access-Control-Allow-Methods": "POST,OPTIONS",
        },
    )


@app.post("/v1/chat/completions")
async def chat_completions(
    request: Request,
    x_api_key: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
) -> Response:
    if not _authorized(x_api_key, authorization):
        return _unauthorized()

    try:
        payload = await request.json()
    except Exception:  # noqa: BLE001
        return JSONResponse(status_code=400, content={"error": "invalid JSON body"})

    try:
        messages = _normalize_messages(payload)
        max_tokens, temperature, top_p = _parse_sampling(payload)
        response_model, bedrock_model_id = _resolve_model(payload.get("model"))
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})

    stream = bool(payload.get("stream"))

    if stream:
        try:
            # Force Bedrock stream open before returning so setup errors are JSON 502.
            generator = _stream_inference(
                messages,
                max_tokens,
                temperature,
                top_p,
                response_model,
                bedrock_model_id,
            )
            first = next(generator)
        except Exception as exc:  # noqa: BLE001
            return _bedrock_error(exc)

        def event_stream() -> Iterator[str]:
            yield first
            yield from generator

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    try:
        inferred = _run_inference(
            messages, max_tokens, temperature, top_p, bedrock_model_id
        )
    except Exception as exc:  # noqa: BLE001
        return _bedrock_error(exc)

    return JSONResponse(
        content=_openai_completion(response_model, inferred["text"], inferred["usage"])
    )


async def _legacy_infer(
    request: Request,
    x_api_key: str | None,
    authorization: str | None,
) -> Response:
    if not _authorized(x_api_key, authorization):
        return _unauthorized()

    try:
        payload = await request.json()
    except Exception:  # noqa: BLE001
        return JSONResponse(status_code=400, content={"error": "invalid JSON body"})

    try:
        messages = _normalize_messages(payload)
        max_tokens, temperature, top_p = _parse_sampling(payload)
        response_model, bedrock_model_id = _resolve_model(payload.get("model"))
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})

    try:
        inferred = _run_inference(
            messages, max_tokens, temperature, top_p, bedrock_model_id
        )
    except Exception as exc:  # noqa: BLE001
        return _bedrock_error(exc)

    return JSONResponse(
        content=_legacy_infer_response(
            response_model, inferred["text"], inferred["usage"]
        )
    )


@app.post("/")
@app.post("/infer")
async def infer(
    request: Request,
    x_api_key: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
) -> Response:
    return await _legacy_infer(request, x_api_key, authorization)
