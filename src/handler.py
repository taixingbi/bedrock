import json
import os
from typing import Any

import boto3
from botocore.exceptions import ClientError

bedrock = boto3.client("bedrock-runtime")

MODEL_ID = os.environ.get("MODEL_ID", "amazon.nova-lite-v1:0")
API_KEY = os.environ.get("API_KEY", "")


def _response(status: int, body: dict[str, Any]) -> dict[str, Any]:
    return {
        "statusCode": status,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "content-type,x-api-key",
            "Access-Control-Allow-Methods": "POST,OPTIONS",
        },
        "body": json.dumps(body),
    }


def _is_imported_model(model_id: str) -> bool:
    return ":imported-model/" in model_id


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


def _infer_converse(prompt: str, system: Any, max_tokens: int) -> dict[str, Any]:
    converse_args: dict[str, Any] = {
        "modelId": MODEL_ID,
        "messages": [
            {
                "role": "user",
                "content": [{"text": prompt}],
            }
        ],
        "inferenceConfig": {
            "maxTokens": max_tokens,
        },
    }
    if isinstance(system, str) and system.strip():
        converse_args["system"] = [{"text": system}]

    result = bedrock.converse(**converse_args)
    usage = result.get("usage") or {}
    return {
        "text": _extract_converse_text(result),
        "usage": {
            "input_tokens": usage.get("inputTokens", 0),
            "output_tokens": usage.get("outputTokens", 0),
        },
    }


def _infer_invoke_model(prompt: str, system: Any, max_tokens: int) -> dict[str, Any]:
    messages: list[dict[str, str]] = []
    if isinstance(system, str) and system.strip():
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    raw = bedrock.invoke_model(
        modelId=MODEL_ID,
        contentType="application/json",
        accept="application/json",
        body=json.dumps(
            {
                "messages": messages,
                "max_tokens": max_tokens,
            }
        ),
    )
    body = json.loads(raw["body"].read())
    usage = body.get("usage") or {}
    return {
        "text": _extract_invoke_text(body),
        "usage": {
            "input_tokens": usage.get("prompt_tokens")
            or usage.get("inputTokens")
            or usage.get("input_tokens")
            or 0,
            "output_tokens": usage.get("completion_tokens")
            or usage.get("outputTokens")
            or usage.get("output_tokens")
            or 0,
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
    if path not in ("/", "/infer"):
        return _response(404, {"error": "not found"})

    headers = {k.lower(): v for k, v in (event.get("headers") or {}).items()}
    provided_key = headers.get("x-api-key", "")
    if not API_KEY or provided_key != API_KEY:
        return _response(401, {"error": "unauthorized"})

    raw_body = event.get("body") or "{}"
    if event.get("isBase64Encoded"):
        import base64

        raw_body = base64.b64decode(raw_body).decode("utf-8")

    try:
        payload = json.loads(raw_body)
    except json.JSONDecodeError:
        return _response(400, {"error": "invalid JSON body"})

    prompt = payload.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        return _response(400, {"error": "prompt is required"})

    system = payload.get("system")
    max_tokens = payload.get("max_tokens", 512)
    if not isinstance(max_tokens, int) or max_tokens < 1 or max_tokens > 4096:
        return _response(400, {"error": "max_tokens must be an integer between 1 and 4096"})

    try:
        if _is_imported_model(MODEL_ID):
            inferred = _infer_invoke_model(prompt, system, max_tokens)
        else:
            inferred = _infer_converse(prompt, system, max_tokens)
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

    return _response(
        200,
        {
            "text": inferred["text"],
            "model": MODEL_ID,
            "usage": inferred["usage"],
        },
    )
