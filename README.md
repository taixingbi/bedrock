# Bedrock MVP Inference API

Python Lambda with a Function URL that sends prompts to Amazon Bedrock via the boto3 **Converse** API.

## Prerequisites

1. AWS account with permission to create Lambda, IAM roles, and call Bedrock
2. [Model access enabled](https://docs.aws.amazon.com/bedrock/latest/userguide/model-access.html) for marketplace models (default: `amazon.nova-lite-v1:0`), **or** a successfully [imported custom model](https://docs.aws.amazon.com/bedrock/latest/userguide/model-customization-import-model.html)
3. [AWS SAM CLI](https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/install-sam-cli.html) for local builds
4. GitHub repository secrets (Settings → Secrets and variables → Actions):

| Secret | Purpose |
| --- | --- |
| `AWS_ACCESS_KEY_ID` | Deploy credentials |
| `AWS_SECRET_ACCESS_KEY` | Deploy credentials |
| `INFERENCE_API_KEY` | Shared secret clients must send as `x-api-key` |

Optional repository variables:

| Variable | Purpose |
| --- | --- |
| `AWS_REGION` | Region for deploy and Bedrock (defaults to `us-east-1`) |
| `MODEL_ID` | Default Bedrock model ID or **imported model ARN** when the request omits `model` (defaults to `amazon.nova-lite-v1:0`) |
| `MODEL_MAP` | Optional JSON object of request alias → Bedrock ID/ARN (merges with built-in aliases) |

The Lambda talks to Bedrock with its **execution role**, not with the deploy access keys. Bedrock is managed inference — you do not choose a GPU.

## Models

The request `model` field selects which Bedrock backend to call. Built-in aliases:

| Request `model` | Bedrock ID | API |
| --- | --- | --- |
| `claude-sonnet` / `claude-sonnet-5` / `anthropic.claude-sonnet-5` | `anthropic.claude-sonnet-5` | Converse |
| `us.anthropic.claude-sonnet-5` | US geo inference profile | Converse |
| `claude-opus` / `claude-opus-4.5` | `us.anthropic.claude-opus-4-5-20251101-v1:0` | Converse |
| `nova-pro` / `amazon.nova-pro-v1:0` | `amazon.nova-pro-v1:0` | Converse |
| `us.amazon.nova-pro-v1:0` | US geo inference profile | Converse |
| `nova-lite` / `amazon.nova-lite-v1:0` | `amazon.nova-lite-v1:0` | Converse |
| `llama` / `llama3.3` / `llama-3.3-70b` | `us.meta.llama3-3-70b-instruct-v1:0` | Converse |
| `llama4` / `llama4-maverick` | `us.meta.llama4-maverick-17b-instruct-v1:0` | Converse |
| `llama4-scout` | `us.meta.llama4-scout-17b-instruct-v1:0` | Converse |
| `gpt-oss` / `gpt-oss-120b` | `openai.gpt-oss-120b-1:0` | Converse |
| `gpt-oss-20b` | `openai.gpt-oss-20b-1:0` | Converse |
| `gpt-5.5` / `openai.gpt-5.5` | `openai.gpt-5.5` | Mantle Responses |
| `deepseek` / `deepseek.v3.2` | `deepseek.v3.2` | Converse |
| `deepseek-r1` | `us.deepseek.r1-v1:0` | Converse |
| `Qwen/Qwen2.5-7B-Instruct` / `qwen` | deployed `MODEL_ID` when it is an imported-model ARN | InvokeModel |

Raw Bedrock IDs and imported-model ARNs are also accepted. Unknown names return `400`.

Override or add aliases with the `MODEL_MAP` env / repo variable, e.g.:

```json
{"my-qwen":"arn:aws:bedrock:us-east-1:646821141010:imported-model/npkn89zkoiyp"}
```

### Default (marketplace)

`amazon.nova-lite-v1:0` — enable access in the Bedrock console for `us-east-1`, then deploy (or set repo variable `MODEL_ID`).

### Amazon Nova Pro (marketplace)

Enable model access in the Bedrock console, then call:

```json
{"model": "nova-pro", "messages": [{"role": "user", "content": "Hello"}]}
```

Optional catalog manifest:

```bash
./scripts/upload-model-to-s3.sh nova-pro
# → s3://bedrock-models-646821141010/amazon/nova-pro-v1/model-manifest.json
```

| Mode | Bedrock ID |
| --- | --- |
| In-region (`us-east-1`) | `amazon.nova-pro-v1:0` |
| US geo cross-region | `us.amazon.nova-pro-v1:0` |

### Claude Sonnet (marketplace)

Anthropic does not publish Claude weights — no `hf download` / Custom Model Import. Enable model access in the Bedrock console, then call:

```json
{"model": "claude-sonnet-5", "messages": [{"role": "user", "content": "Hello"}]}
```

Optional catalog manifest in the shared bucket:

```bash
./scripts/upload-model-to-s3.sh claude-sonnet
# → s3://bedrock-models-646821141010/anthropic/claude-sonnet-5/model-manifest.json
```

| Mode | Bedrock ID |
| --- | --- |
| In-region (`us-east-1`) | `anthropic.claude-sonnet-5` |
| US geo cross-region | `us.anthropic.claude-sonnet-5` |
| Global | `global.anthropic.claude-sonnet-5` |

Submit the Anthropic use-case form in the Bedrock console before first invoke.

### Claude Opus 4.5 (marketplace)

Enable model access, then call (friendly aliases default to the **US geo** inference profile):

```json
{"model": "claude-opus-4.5", "messages": [{"role": "user", "content": "Hello"}]}
```

```bash
./scripts/upload-model-to-s3.sh claude-opus
# → s3://bedrock-models-646821141010/anthropic/claude-opus-4-5/model-manifest.json
```

| Mode | Bedrock ID |
| --- | --- |
| In-region | `anthropic.claude-opus-4-5-20251101-v1:0` |
| US geo cross-region (default alias) | `us.anthropic.claude-opus-4-5-20251101-v1:0` |
| Global | `global.anthropic.claude-opus-4-5-20251101-v1:0` |

### Meta Llama (marketplace)

Enable Meta model access in the Bedrock console. Friendly aliases default to the **US geo inference profile** (required for on-demand on many Llama IDs):

```json
{"model": "llama", "messages": [{"role": "user", "content": "Hello"}]}
```

| Alias | Bedrock ID |
| --- | --- |
| `llama` / `llama3.3` | `us.meta.llama3-3-70b-instruct-v1:0` |
| `llama4` / `llama4-maverick` | `us.meta.llama4-maverick-17b-instruct-v1:0` |
| `llama4-scout` | `us.meta.llama4-scout-17b-instruct-v1:0` |

```bash
./scripts/upload-model-to-s3.sh llama
# → s3://bedrock-models-646821141010/meta/llama3-3-70b-instruct/model-manifest.json
```

### OpenAI GPT-OSS (marketplace)

Bedrock-hosted OpenAI open-weight models (not ChatGPT API keys). Enable access, then:

```json
{"model": "gpt-oss", "messages": [{"role": "user", "content": "Hello"}]}
```

| Alias | Bedrock ID |
| --- | --- |
| `gpt-oss` / `gpt-oss-120b` | `openai.gpt-oss-120b-1:0` |
| `gpt-oss-20b` | `openai.gpt-oss-20b-1:0` |

```bash
./scripts/upload-model-to-s3.sh gpt-oss
# → s3://bedrock-models-646821141010/openai/gpt-oss-120b/model-manifest.json
```

### OpenAI GPT-5.5 (marketplace)

GPT-5.5 on Bedrock is **Responses-API only** via `bedrock-mantle` (not Converse). The Lambda signs Mantle requests with SigV4. Enable access in `us-east-1` or `us-east-2`, then:

```json
{"model": "gpt-5.5", "messages": [{"role": "user", "content": "Hello"}]}
```

Omit `temperature` / `top_p` — GPT-5.x rejects them. Optional env: `BEDROCK_MANTLE_REGION`, `BEDROCK_MANTLE_REASONING_EFFORT` (default `minimal`).

```bash
./scripts/upload-model-to-s3.sh gpt-5.5
# → s3://bedrock-models-646821141010/openai/gpt-5.5/model-manifest.json
```

### DeepSeek (marketplace)

Enable DeepSeek access, then:

```json
{"model": "deepseek", "messages": [{"role": "user", "content": "Hello"}]}
```

| Alias | Bedrock ID |
| --- | --- |
| `deepseek` / `deepseek-v3.2` | `deepseek.v3.2` |
| `deepseek-r1` | `us.deepseek.r1-v1:0` |

```bash
./scripts/upload-model-to-s3.sh deepseek
# → s3://bedrock-models-646821141010/deepseek/deepseek-v3.2/model-manifest.json
```

### Custom import (e.g. Qwen2.5)

Qwen2.5 is not a built-in Bedrock marketplace ID. Download Hugging Face weights, upload to S3, then [Custom Model Import](https://docs.aws.amazon.com/bedrock/latest/userguide/model-customization-import-model.html). Set `MODEL_ID` to the **imported model ARN** (also used for the `Qwen/Qwen2.5-7B-Instruct` alias).

Shared models bucket (`us-east-1`):

```text
s3://bedrock-models-646821141010/
  qwen/Qwen2.5-7B-Instruct/   ← config.json must live here
  anthropic/claude-sonnet-5/  ← marketplace manifest (no HF weights)
  anthropic/claude-opus-4-5/  ← marketplace manifest (no HF weights)
  amazon/nova-pro-v1/         ← marketplace manifest (no HF weights)
  meta/llama3-3-70b-instruct/ ← marketplace manifest (no HF weights)
  openai/gpt-oss-120b/        ← marketplace manifest (no HF weights)
  openai/gpt-5.5/             ← marketplace manifest (mantle Responses)
  deepseek/deepseek-v3.2/     ← marketplace manifest (no HF weights)
```

Marketplace Claude/Nova/Meta/OpenAI/DeepSeek models are enabled in the Bedrock console — they are not stored as HF weights in this bucket.

#### 1. Download and upload Qwen2.5-7B-Instruct

```bash
# needs: pip install huggingface_hub  (provides `hf`)
./scripts/upload-model-to-s3.sh qwen

# or sync an existing checkout:
./scripts/upload-model-to-s3.sh qwen --local ./Qwen2.5-7B-Instruct
```

(~15 GB download; keep model dirs and `.venv-hf` out of git.)

#### 2. IAM role for import

Role: `arn:aws:iam::646821141010:role/BedrockModelImportRole`

Trust policy must allow `bedrock.amazonaws.com` to assume the role. Permissions need `s3:GetObject` / `s3:ListBucket` on `bedrock-models-646821141010`.

#### 3. Create import job

Also raise Service Quotas → Bedrock → **Concurrent model import jobs** if needed (request > current value).

```bash
aws bedrock create-model-import-job \
  --region us-east-1 \
  --job-name qwen25-7b-instruct-import-3 \
  --imported-model-name qwen25-7b-instruct \
  --role-arn arn:aws:iam::646821141010:role/BedrockModelImportRole \
  --model-data-source '{"s3DataSource":{"s3Uri":"s3://bedrock-models-646821141010/qwen/Qwen2.5-7B-Instruct/"}}'
```

Poll status, then list the ARN:

```bash
aws bedrock get-model-import-job \
  --region us-east-1 \
  --job-identifier JOB_ARN

aws bedrock list-imported-models --region us-east-1
```

Set GitHub variable `MODEL_ID` to the imported model ARN and redeploy.

Current imported model (`Qwen2.5-7B-Instruct`):

```text
arn:aws:bedrock:us-east-1:646821141010:imported-model/npkn89zkoiyp
```

This is set as the `MODEL_ID` repository variable (default + `Qwen/...` alias). Claude Sonnet works in the same deploy via built-in aliases — no need to swap `MODEL_ID`.

The handler picks the Bedrock API per resolved model:

- Marketplace models (Claude Sonnet, Nova, …) → **Converse**
- Imported models (`:imported-model/` ARN, e.g. Qwen2.5) → **InvokeModel** with an OpenAI-compatible `messages` body

Qwen2.5 Custom Model Import does not support Converse; InvokeModel is required.

## API

### OpenAI-compatible (preferred)

`POST` `/v1/chat/completions`

```json
{
  "model": "Qwen/Qwen2.5-7B-Instruct",
  "messages": [
    {"role": "system", "content": "optional system prompt"},
    {"role": "user", "content": "Hello"}
  ],
  "max_tokens": 512,
  "temperature": 0,
  "top_p": 1.0
}
```

Headers (either works):

- `Authorization: Bearer <INFERENCE_API_KEY>`
- `x-api-key: <INFERENCE_API_KEY>`

Success (OpenAI chat.completion shape):

```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "created": 0,
  "model": "Qwen/Qwen2.5-7B-Instruct",
  "choices": [
    {
      "index": 0,
      "message": {"role": "assistant", "content": "..."},
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0
  }
}
```

The `model` field selects the Bedrock backend (see [Models](#models)); the same name is echoed in the response.

Set `"stream": true` to receive OpenAI SSE (`text/event-stream`) chunks (`chat.completion.chunk` then `data: [DONE]`). Streaming uses Lambda Function URL `RESPONSE_STREAM` plus Bedrock `InvokeModelWithResponseStream` / `ConverseStream`.

## Deploy

Push to `main` or run the **Deploy** workflow manually. Stack name: `bedrock-inference-mvp` (region defaults to `us-east-1`).

After deploy, get the Function URL (include `--region` if `aws configure` has no default):

```bash
aws cloudformation describe-stacks \
  --region us-east-1 \
  --stack-name bedrock-inference-mvp \
  --query "Stacks[0].Outputs[?OutputKey=='InferenceFunctionUrl'].OutputValue" \
  --output text
```

Or in the AWS Console: CloudFormation → stack `bedrock-inference-mvp` → **Outputs** → `InferenceFunctionUrl` (or Lambda → `bedrock-inference-mvp` → **Configuration** → **Function URL**).

### Manual deploy

```bash
sam build
sam deploy \
  --region us-east-1 \
  --parameter-overrides \
    ModelId=arn:aws:bedrock:us-east-1:646821141010:imported-model/npkn89zkoiyp \
    ApiKey='your-shared-secret'
```

## Call the API

Qwen (imported):

```bash
FUNCTION_URL=$(aws cloudformation describe-stacks \
  --region us-east-1 \
  --stack-name bedrock-inference-mvp \
  --query "Stacks[0].Outputs[?OutputKey=='InferenceFunctionUrl'].OutputValue" \
  --output text)
INFERENCE_API_KEY='1234'

curl -sS -N -X POST "${FUNCTION_URL}v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${INFERENCE_API_KEY}" \
  -d '{
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "messages": [{"role": "user", "content": "Say hello in one short sentence."}],
    "max_tokens": 64,
    "temperature": 0,
    "top_p": 1.0,
    "stream": true
  }'
```

Claude Sonnet (marketplace):

```bash
curl -sS -X POST "${FUNCTION_URL}v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${INFERENCE_API_KEY}" \
  -d '{
    "model": "claude-sonnet-5",
    "messages": [{"role": "user", "content": "Say hello in one short sentence."}],
    "max_tokens": 64,
    "temperature": 0
  }' | jq '{model, answer: .choices[0].message.content, usage}'
```

Amazon Nova Pro (marketplace):

```bash
curl -sS -X POST "${FUNCTION_URL}v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${INFERENCE_API_KEY}" \
  -d '{
    "model": "nova-pro",
    "messages": [{"role": "user", "content": "Say hello in one short sentence."}],
    "max_tokens": 64,
    "temperature": 0
  }' | jq '{model, answer: .choices[0].message.content, usage}'
```

Meta Llama 3.3 (marketplace):

```bash
curl -sS -X POST "${FUNCTION_URL}v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${INFERENCE_API_KEY}" \
  -d '{
    "model": "llama",
    "messages": [{"role": "user", "content": "Say hello in one short sentence."}],
    "max_tokens": 64,
    "temperature": 0
  }' | jq '{model, answer: .choices[0].message.content, usage}'
```

OpenAI GPT-OSS (marketplace):

```bash
curl -sS -X POST "${FUNCTION_URL}v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${INFERENCE_API_KEY}" \
  -d '{
    "model": "gpt-oss",
    "messages": [{"role": "user", "content": "Say hello in one short sentence."}],
    "max_tokens": 64,
    "temperature": 0
  }' | jq '{model, answer: .choices[0].message.content, usage}'
```

Claude Opus 4.5 (marketplace):

```bash
curl -sS -X POST "${FUNCTION_URL}v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${INFERENCE_API_KEY}" \
  -d '{
    "model": "claude-opus-4.5",
    "messages": [{"role": "user", "content": "Say hello in one short sentence."}],
    "max_tokens": 64,
    "temperature": 0
  }' | jq '{model, answer: .choices[0].message.content, usage}'
```

DeepSeek V3.2 (marketplace):

```bash
curl -sS -X POST "${FUNCTION_URL}v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${INFERENCE_API_KEY}" \
  -d '{
    "model": "deepseek",
    "messages": [{"role": "user", "content": "Say hello in one short sentence."}],
    "max_tokens": 64,
    "temperature": 0
  }' | jq '{model, answer: .choices[0].message.content, usage}'
```

OpenAI GPT-5.5 (marketplace, Mantle):

```bash
curl -sS -X POST "${FUNCTION_URL}v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${INFERENCE_API_KEY}" \
  -d '{
    "model": "gpt-5.5",
    "messages": [{"role": "user", "content": "Say hello in one short sentence."}],
    "max_tokens": 64
  }' | jq '{model, answer: .choices[0].message.content, usage}'
```

## Local invoke

Run the FastAPI app locally (uses your AWS credentials for Bedrock):

```bash
export API_KEY=local-dev-key
export MODEL_ID='arn:aws:bedrock:us-east-1:646821141010:imported-model/npkn89zkoiyp'
pip install -r src/requirements.txt
uvicorn app:app --app-dir src --host 127.0.0.1 --port 8080
```

Then:

```bash
curl -sS -N -X POST "http://127.0.0.1:8080/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer local-dev-key" \
  -d '{
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "messages": [{"role": "user", "content": "Say hello in one short sentence."}],
    "max_tokens": 64,
    "temperature": 0,
    "stream": true
  }'
```