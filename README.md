# Bedrock MVP Inference API

Python Lambda with a Function URL that sends prompts to Amazon Bedrock via the boto3 **Converse** API.

## Prerequisites

1. AWS account with permission to create Lambda, IAM roles, and call Bedrock
2. [Model access enabled](https://docs.aws.amazon.com/bedrock/latest/userguide/model-access.html) for your chosen model in the target region (default: `amazon.nova-lite-v1:0`)
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
| `MODEL_ID` | Bedrock model ID (defaults to `amazon.nova-lite-v1:0`) |

The Lambda talks to Bedrock with its **execution role**, not with the deploy access keys.

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

Push to `main` or run the **Deploy** workflow manually. Stack name: `bedrock-inference-mvp`.

After deploy, read the Function URL:

```bash
aws cloudformation describe-stacks \
  --stack-name bedrock-inference-mvp \
  --query "Stacks[0].Outputs[?OutputKey=='InferenceFunctionUrl'].OutputValue" \
  --output text
```

### Manual deploy

```bash
sam build
sam deploy \
  --parameter-overrides \
    ModelId=amazon.nova-lite-v1:0 \
    ApiKey='your-shared-secret'
```

## Call the API

```bash
curl -sS -X POST "$FUNCTION_URL/infer" \
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
