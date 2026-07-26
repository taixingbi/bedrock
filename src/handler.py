import base64
import json
import os
import time
import uuid
from typing import Any

import boto3
from botocore.exceptions import ClientError

bedrock = boto3.client("bedrock-runtime")

MODEL_ID = os.environ.get("MODEL_ID", "amazon.nova-lite-v1:0")
API_KEY = os.environ.get("API_KEY", "")

CORS_HEADERS = {
    "Content-Type": "application/json",
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "content-type,x-api-key,authorization",
    "Access-Control-Allow-Methods": "POST,OPTIONS",
}


def _response(status: int, body: dict[str, Any]) -> dict[str, Any]:
    return {
        "statusCode": status,
        "headers": CORS_HEADERS,
        "body": json.dumps(body),
    }


def _is_imported_model(model_id: str) -> bool:
    return ":imported-model/" in model_id


def _authorized(headers: dict[str, str]) -> bool:
    if not API_KEY:
        return False
    if headers.get("x-api-key", "") == API_KEY:
        return True
    auth = headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip() == API_KEY
    return False


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
    """Accept OpenAI messages, or legacy {prompt, system}."""
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

    normalized = []
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


def _infer_converse(
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float | None,
    top_p: float | None,
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
        "modelId": MODEL_ID,
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
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "messages": messages,
        "max_tokens": max_tokens,
    }
    if temperature is not None:
        body["temperature"] = temperature
    if top_p is not None:
        body["top_p"] = top_p

    raw = bedrock.invoke_model(
        modelId=MODEL_ID,
        contentType="application/json",
        accept="application/json",
        body=json.dumps(body),
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


def _openai_completion(
    model: str,
    text: str,
    usage: dict[str, int],
) -> dict[str, Any]:
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


def handler(event: dict[str, Any], _context: Any) -> dict[str, Any]:
    request_context = event.get("requestContext", {})
    http = request_context.get("http", {})
    method = (http.get("method") or event.get("httpMethod") or "POST").upper()

    if method == "OPTIONS":
        return _response(204, {})

    if method != "POST":
        return _response(405, {"error": "method not allowed"})

    path = http.get("path") or event.get("rawPath") or "/"
    path = path.rstrip("/") or "/"
    openai_path = path in ("/v1/chat/completions",)
    legacy_path = path in ("/", "/infer")
    if not openai_path and not legacy_path:
        return _response(404, {"error": "not found"})

    headers = {k.lower(): v for k, v in (event.get("headers") or {}).items()}
    if not _authorized(headers):
        return _response(401, {"error": "unauthorized"})

    raw_body = event.get("body") or "{}"
    if event.get("isBase64Encoded"):
        raw_body = base64.b64decode(raw_body).decode("utf-8")

    try:
        payload = json.loads(raw_body)
    except json.JSONDecodeError:
        return _response(400, {"error": "invalid JSON body"})

    try:
        messages = _normalize_messages(payload)
        max_tokens, temperature, top_p = _parse_sampling(payload)
    except ValueError as exc:
        return _response(400, {"error": str(exc)})

    request_model = payload.get("model")
    response_model = request_model if isinstance(request_model, str) and request_model else MODEL_ID

    try:
        if _is_imported_model(MODEL_ID):
            inferred = _infer_invoke_model(messages, max_tokens, temperature, top_p)
        else:
            inferred = _infer_converse(messages, max_tokens, temperature, top_p)
    except ClientError as exc:
        return _response(
            502,
            {
                "error": "bedrock request failed",
                "detail": exc.response.get("Error", {}).get("Message", str(exc)),
            },
        )
    except Exception as exc:  # noqa: BLE001
        return _response(502, {"error": "bedrock request failed", "detail": str(exc)})

    if openai_path:
        return _response(
            200,
            _openai_completion(response_model, inferred["text"], inferred["usage"]),
        )

    return _response(
        200,
        _legacy_infer_response(response_model, inferred["text"], inferred["usage"]),
    )
