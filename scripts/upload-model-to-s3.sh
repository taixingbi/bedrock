#!/usr/bin/env bash
# Upload / register a model in the shared Bedrock models bucket.
#
# Usage:
#   ./scripts/upload-model-to-s3.sh claude-sonnet
#   ./scripts/upload-model-to-s3.sh claude-opus
#   ./scripts/upload-model-to-s3.sh nova-pro
#   ./scripts/upload-model-to-s3.sh llama
#   ./scripts/upload-model-to-s3.sh gpt-oss
#   ./scripts/upload-model-to-s3.sh gpt-5.5
#   ./scripts/upload-model-to-s3.sh deepseek
#   ./scripts/upload-model-to-s3.sh qwen
#   ./scripts/upload-model-to-s3.sh qwen --local ./Qwen2.5-7B-Instruct
#
# Env overrides:
#   BUCKET      default s3://bedrock-models-646821141010
#   AWS_REGION  default us-east-1
set -euo pipefail

BUCKET="${BUCKET:-s3://bedrock-models-646821141010}"
REGION="${AWS_REGION:-us-east-1}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
  cat <<'EOF'
Usage: ./scripts/upload-model-to-s3.sh <model> [options]

Models:
  claude-sonnet   Register marketplace manifest (no HF weights)
  claude-opus     Register Claude Opus 4.5 marketplace manifest
  nova-pro        Register marketplace manifest (no HF weights)
  llama           Register Meta Llama 3.3 70B marketplace manifest
  gpt-oss         Register OpenAI GPT-OSS 120B marketplace manifest
  gpt-5.5         Register OpenAI GPT-5.5 marketplace manifest (mantle)
  deepseek        Register DeepSeek V3.2 marketplace manifest
  qwen            Download Qwen2.5-7B-Instruct (unless --local) and s3 sync

Options (qwen):
  --local DIR     Sync existing local weights instead of hf download

Env:
  BUCKET, AWS_REGION
  MODEL_ID, MODEL_NAME (marketplace models)
EOF
}

die() { echo "error: $*" >&2; exit 1; }

require_aws() {
  command -v aws >/dev/null || die "aws CLI required"
}

# Register a Bedrock marketplace model catalog entry (no weights).
# Args: display_name provider prefix_name model_id request_aliases_csv [us_profile] [global_profile|""]
upload_marketplace_manifest() {
  require_aws

  local display_name="$1"
  local provider="$2"
  local model_name="$3"
  local model_id="$4"
  local aliases="$5"
  local us_profile="${6:-}"
  local global_profile="${7-}"
  local prefix="${provider}/${model_name}"

  local tmp
  tmp="$(mktemp -d)"
  trap 'rm -rf "$tmp"' RETURN

  local profiles
  if [[ -n "${us_profile}" && -n "${global_profile}" ]]; then
    profiles=$(cat <<EOF
  "inference_profile_ids": {
    "in_region": "${model_id}",
    "us": "${us_profile}",
    "global": "${global_profile}"
  },
EOF
)
  elif [[ -n "${us_profile}" ]]; then
    profiles=$(cat <<EOF
  "inference_profile_ids": {
    "in_region": "${model_id}",
    "us": "${us_profile}"
  },
EOF
)
  else
    profiles=$(cat <<EOF
  "inference_profile_ids": {
    "in_region": "${model_id}"
  },
EOF
)
  fi

  local manifest="${tmp}/model-manifest.json"
  cat >"${manifest}" <<EOF
{
  "name": "${display_name}",
  "provider": "${provider}",
  "type": "bedrock-marketplace",
  "model_id": "${model_id}",
${profiles}
  "region": "${REGION}",
  "s3_prefix": "${BUCKET}/${prefix}/",
  "note": "Marketplace model — enable access in the Bedrock console. Weights are not stored in this bucket. Request model aliases: ${aliases}."
}
EOF

  local dest="${BUCKET}/${prefix}/model-manifest.json"
  echo "Uploading manifest → ${dest}"
  aws s3 cp "${manifest}" "${dest}" --region "${REGION}"

  echo
  echo "Done. Catalog entry: ${BUCKET}/${prefix}/"
  echo
  echo "Next:"
  echo "  1. Bedrock console → Model access → enable ${display_name} (${model_id})"
  echo "  2. Redeploy — request \"model\": \"$(echo "${aliases}" | cut -d, -f1 | tr -d ' ')\" (Converse; no Custom Model Import)"
  echo
  aws s3 ls "${BUCKET}/${prefix}/" --region "${REGION}"
}

upload_claude_sonnet() {
  local model_name="${MODEL_NAME:-claude-sonnet-5}"
  local model_id="${MODEL_ID:-anthropic.claude-sonnet-5}"
  upload_marketplace_manifest \
    "Claude Sonnet" \
    "anthropic" \
    "${model_name}" \
    "${model_id}" \
    "claude-sonnet, claude-sonnet-5, anthropic.claude-sonnet-5" \
    "${US_PROFILE:-us.${model_id}}" \
    "${GLOBAL_PROFILE:-global.${model_id}}"
}

upload_claude_opus() {
  local model_name="${MODEL_NAME:-claude-opus-4-5}"
  local model_id="${MODEL_ID:-anthropic.claude-opus-4-5-20251101-v1:0}"
  upload_marketplace_manifest \
    "Claude Opus 4.5" \
    "anthropic" \
    "${model_name}" \
    "${model_id}" \
    "claude-opus, claude-opus-4.5, claude-opus-4-5, us.anthropic.claude-opus-4-5-20251101-v1:0" \
    "${US_PROFILE:-us.${model_id}}" \
    "${GLOBAL_PROFILE:-global.${model_id}}"
}

upload_nova_pro() {
  local model_name="${MODEL_NAME:-nova-pro-v1}"
  local model_id="${MODEL_ID:-amazon.nova-pro-v1:0}"
  upload_marketplace_manifest \
    "Amazon Nova Pro" \
    "amazon" \
    "${model_name}" \
    "${model_id}" \
    "nova-pro, amazon.nova-pro-v1:0" \
    "${US_PROFILE:-us.${model_id}}" \
    ""
}

upload_llama() {
  local model_name="${MODEL_NAME:-llama3-3-70b-instruct}"
  local model_id="${MODEL_ID:-meta.llama3-3-70b-instruct-v1:0}"
  upload_marketplace_manifest \
    "Meta Llama 3.3 70B Instruct" \
    "meta" \
    "${model_name}" \
    "${model_id}" \
    "llama, llama3.3, llama-3.3-70b, us.meta.llama3-3-70b-instruct-v1:0" \
    "${US_PROFILE:-us.${model_id}}" \
    ""
}

upload_gpt_oss() {
  local model_name="${MODEL_NAME:-gpt-oss-120b}"
  local model_id="${MODEL_ID:-openai.gpt-oss-120b-1:0}"
  upload_marketplace_manifest \
    "OpenAI GPT-OSS 120B" \
    "openai" \
    "${model_name}" \
    "${model_id}" \
    "gpt-oss, gpt-oss-120b, openai.gpt-oss-120b-1:0" \
    "" \
    ""
}

upload_gpt_55() {
  local model_name="${MODEL_NAME:-gpt-5.5}"
  local model_id="${MODEL_ID:-openai.gpt-5.5}"
  upload_marketplace_manifest \
    "OpenAI GPT-5.5" \
    "openai" \
    "${model_name}" \
    "${model_id}" \
    "gpt-5.5, gpt-5-5, openai.gpt-5.5" \
    "" \
    ""
}

upload_deepseek() {
  local model_name="${MODEL_NAME:-deepseek-v3.2}"
  local model_id="${MODEL_ID:-deepseek.v3.2}"
  upload_marketplace_manifest \
    "DeepSeek V3.2" \
    "deepseek" \
    "${model_name}" \
    "${model_id}" \
    "deepseek, deepseek-v3.2, deepseek.v3.2" \
    "" \
    ""
}

upload_qwen() {
  require_aws

  local local_dir=""
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --local)
        [[ $# -ge 2 ]] || die "--local requires a directory"
        local_dir="$2"
        shift 2
        ;;
      *)
        die "unknown option for qwen: $1"
        ;;
    esac
  done

  local hf_repo="Qwen/Qwen2.5-7B-Instruct"
  local prefix="qwen/Qwen2.5-7B-Instruct"
  local dest="${BUCKET}/${prefix}/"

  if [[ -z "${local_dir}" ]]; then
    local_dir="${ROOT}/Qwen2.5-7B-Instruct"
    command -v hf >/dev/null || die "hf CLI required (pip install huggingface_hub), or pass --local DIR"
    echo "Downloading ${hf_repo} → ${local_dir}"
    hf download "${hf_repo}" --local-dir "${local_dir}"
  fi

  [[ -f "${local_dir}/config.json" ]] || die "missing ${local_dir}/config.json"

  echo "Syncing ${local_dir} → ${dest}"
  aws s3 sync "${local_dir}" "${dest}" \
    --region "${REGION}" \
    --exclude ".cache/*"

  echo
  echo "Done. Weights at ${dest}"
  echo "Create a Custom Model Import job pointing at that s3Uri (see README)."
}

MODEL="${1:-}"
[[ -n "${MODEL}" ]] || { usage; exit 1; }
shift || true

case "${MODEL}" in
  -h|--help|help)
    usage
    ;;
  claude-sonnet|claude-sonnet-5)
    upload_claude_sonnet
    ;;
  claude-opus|claude-opus-4.5|claude-opus-4-5)
    upload_claude_opus
    ;;
  nova-pro|nova-pro-v1|amazon.nova-pro-v1:0)
    upload_nova_pro
    ;;
  llama|llama3.3|llama-3.3-70b|meta.llama3-3-70b-instruct-v1:0)
    upload_llama
    ;;
  gpt-oss|gpt-oss-120b|openai.gpt-oss-120b-1:0)
    upload_gpt_oss
    ;;
  gpt-5.5|gpt-5-5|openai.gpt-5.5)
    upload_gpt_55
    ;;
  deepseek|deepseek-v3.2|deepseek.v3.2)
    upload_deepseek
    ;;
  qwen|Qwen2.5-7B-Instruct|qwen2.5-7b-instruct)
    upload_qwen "$@"
    ;;
  *)
    die "unknown model '${MODEL}' (try: claude-sonnet, claude-opus, nova-pro, llama, gpt-oss, gpt-5.5, deepseek, qwen)"
    ;;
esac
