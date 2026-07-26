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
| `MODEL_ID` | Bedrock model ID or **imported model ARN** (defaults to `amazon.nova-lite-v1:0`) |

The Lambda talks to Bedrock with its **execution role**, not with the deploy access keys. Bedrock is managed inference — you do not choose a GPU.

## Models

### Default (marketplace)

`amazon.nova-lite-v1:0` — enable access in the Bedrock console for `us-east-1`, then deploy (or set repo variable `MODEL_ID`).

### Custom import (e.g. Qwen2.5)

Qwen2.5 is not a built-in Bedrock marketplace ID. Download Hugging Face weights, upload to S3, then [Custom Model Import](https://docs.aws.amazon.com/bedrock/latest/userguide/model-customization-import-model.html). Set `MODEL_ID` to the **imported model ARN**.

Shared models bucket (`us-east-1`):

```text
s3://bedrock-models-646821141010/
  qwen/Qwen2.5-7B-Instruct/   ← config.json must live here
  anthropic/                  ← future open/custom weights
  meta/                       ← future
```

Marketplace Claude/Nova models are enabled in the Bedrock console — they are not stored as HF weights in this bucket.

#### 1. Download and upload Qwen2.5-7B-Instruct

```bash
python3 -m venv .venv-hf
source .venv-hf/bin/activate
python -m pip install --upgrade pip huggingface_hub

hf download Qwen/Qwen2.5-7B-Instruct --local-dir ./Qwen2.5-7B-Instruct
ls ./Qwen2.5-7B-Instruct/config.json

aws s3 sync ./Qwen2.5-7B-Instruct \
  s3://bedrock-models-646821141010/qwen/Qwen2.5-7B-Instruct/ \
  --region us-east-1 \
  --exclude ".cache/*"

deactivate
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

Set GitHub variable `MODEL_ID` to the imported model ARN and redeploy. If Converse fails for the imported model, the handler may need `InvokeModel` instead (not covered by this MVP).

Current imported model (`Qwen2.5-7B-Instruct`):

```text
arn:aws:bedrock:us-east-1:646821141010:imported-model/npkn89zkoiyp
```

This is set as the `MODEL_ID` repository variable, so deploys use Qwen2.5 by default. To switch back to Nova, set `MODEL_ID=amazon.nova-lite-v1:0` and redeploy.

## API

`POST` `/` or `/infer`

```json
{
  "prompt": "Hello",
  "system": "optional system prompt",
  "max_tokens": 512
}
```

Headers:

- `Content-Type: application/json`
- `x-api-key: <INFERENCE_API_KEY>`

Success:

```json
{
  "text": "...",
  "model": "amazon.nova-lite-v1:0",
  "usage": {
    "input_tokens": 0,
    "output_tokens": 0
  }
}
```

Errors: `400` bad body, `401` missing/wrong key, `404` unknown path, `405` method, `502` Bedrock failure.

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
    ModelId=amazon.nova-lite-v1:0 \
    ApiKey='your-shared-secret'
```

## Call the API

```bash
export FUNCTION_URL='https://xxxx.lambda-url.us-east-1.on.amazonaws.com/'
export INFERENCE_API_KEY='your-shared-secret'

curl -sS -X POST "${FUNCTION_URL}infer" \
  -H "Content-Type: application/json" \
  -H "x-api-key: $INFERENCE_API_KEY" \
  -d '{"prompt":"Say hello in one short sentence.","max_tokens":64}'
```

## Local invoke

Set `API_KEY=local-dev-key` (matches `events/infer.json`), then:

```bash
sam build
sam local invoke InferenceFunction --event events/infer.json
```
