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


def _extract_text(converse_response: dict[str, Any]) -> str:
    parts: list[str] = []
    message = converse_response.get("output", {}).get("message", {})
    for block in message.get("content", []):
        text = block.get("text")
        if text:
            parts.append(text)
    return "".join(parts)


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

    try:
        result = bedrock.converse(**converse_args)
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

    usage = result.get("usage") or {}
    return _response(
        200,
        {
            "text": _extract_text(result),
            "model": MODEL_ID,
            "usage": {
                "input_tokens": usage.get("inputTokens", 0),
                "output_tokens": usage.get("outputTokens", 0),
            },
        },
    )
