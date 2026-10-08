#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATH_FILE="${SEER_PATHS_FILE:-${ROOT_DIR}/configs/paths.sh}"

if [[ ! -f "${PATH_FILE}" ]]; then
  echo "Missing ${PATH_FILE}. Copy configs/paths.example.sh to configs/paths.sh and edit it." >&2
  exit 2
fi

# shellcheck source=/dev/null
source "${PATH_FILE}"
cd "${ROOT_DIR}"

dataset="${1:-}"
stage="${2:-}"
if [[ "${dataset}" != "ravdess" && "${dataset}" != "esd" ]]; then
  echo "Usage: bash run.sh {ravdess|esd} {preprocess|build-bank|train-seer|enhance|evaluate-ser|evaluate-wer}" >&2
  exit 2
fi

python_bin="${SEER_PYTHON:-python}"
device="${SEER_DEVICE:-cuda}"

required=(
  SEER_MAESTRO_ROOT SEER_EMOTION_DIARIZATION_MODEL SEER_SPEAKER_MODEL
  SEER_EMOTION2VEC_MODEL SEER_FEATURE_CACHE
)
for name in "${required[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "${name} is not set in ${PATH_FILE}" >&2
    exit 2
  fi
done

if [[ "${dataset}" == "ravdess" ]]; then
  raw="${SEER_RAVDESS_RAW}"
  data_root="${SEER_RAVDESS_DATA}"
  bank_root="${SEER_RAVDESS_BANK}"
  retriever_root="${SEER_RAVDESS_RETRIEVER}"
  enhanced_root="${SEER_RAVDESS_OUTPUT}"
  selector_config="${ROOT_DIR}/configs/seer/ravdess.json"
  maestro_checkpoint="${SEER_RAVDESS_MAESTRO_CHECKPOINT}"
  maestro_config="${SEER_RAVDESS_MAESTRO_CONFIG}"
  evaluation_root="${SEER_RAVDESS_EVALUATION:-${SEER_RUN_ROOT}/ravdess/evaluation}"
  source_options=(--source-intensity normal)
else
  raw="${SEER_ESD_RAW}"
  data_root="${SEER_ESD_DATA}"
  bank_root="${SEER_ESD_BANK}"
  retriever_root="${SEER_ESD_RETRIEVER}"
  enhanced_root="${SEER_ESD_OUTPUT}"
  selector_config="${ROOT_DIR}/configs/seer/esd.json"
  maestro_checkpoint="${SEER_ESD_MAESTRO_CHECKPOINT}"
  maestro_config="${SEER_ESD_MAESTRO_CONFIG}"
  evaluation_root="${SEER_ESD_EVALUATION:-${SEER_RUN_ROOT}/esd-en/evaluation}"
  source_options=()
fi

case "${stage}" in
  preprocess)
    if [[ "${dataset}" == "ravdess" ]]; then
      "${python_bin}" -m data.ravdess_preprocess \
        --root "${raw}" --output "${data_root}" --fold 0 \
        --maestro-root "${SEER_MAESTRO_ROOT}" \
        --emotion-diarization-model "${SEER_EMOTION_DIARIZATION_MODEL}" \
        --speaker-model "${SEER_SPEAKER_MODEL}" \
        --emotion2vec-model "${SEER_EMOTION2VEC_MODEL}" \
        --feature-cache "${SEER_FEATURE_CACHE}" --device "${device}"
    else
      "${python_bin}" -m data.esd_preprocess \
        --root "${raw}" --output "${data_root}" --language en \
        --maestro-root "${SEER_MAESTRO_ROOT}" \
        --emotion-diarization-model "${SEER_EMOTION_DIARIZATION_MODEL}" \
        --speaker-model "${SEER_SPEAKER_MODEL}" \
        --emotion2vec-model "${SEER_EMOTION2VEC_MODEL}" \
        --feature-cache "${SEER_FEATURE_CACHE}" --device "${device}"
    fi
    ;;
  build-bank)
    "${python_bin}" -m data.reference_bank \
      --config "${selector_config}" --data-root "${data_root}" \
      --output "${bank_root}" --emotion2vec-model "${SEER_EMOTION2VEC_MODEL}" \
      --device "${device}"
    ;;
  train-seer)
    "${python_bin}" -m seer.train \
      --config "${selector_config}" --data-root "${data_root}" \
      --bank-root "${bank_root}" --maestro-root "${SEER_MAESTRO_ROOT}" \
      --maestro-checkpoint "${maestro_checkpoint}" \
      --maestro-config "${maestro_config}" \
      --emotion2vec-model "${SEER_EMOTION2VEC_MODEL}" \
      --output "${retriever_root}" --reward-device "${device}"
    ;;
  enhance)
    "${python_bin}" -m seer.enhance \
      --config "${selector_config}" --data-root "${data_root}" \
      --manifest "${data_root}/test.csv" --bank-root "${bank_root}" \
      --checkpoint "${retriever_root}/best.pt" \
      --maestro-root "${SEER_MAESTRO_ROOT}" \
      --maestro-checkpoint "${maestro_checkpoint}" \
      --maestro-config "${maestro_config}" \
      --output "${enhanced_root}" --evc-device "${device}" \
      "${source_options[@]}"
    ;;
  evaluate-ser)
    "${python_bin}" -m seer.evaluate \
      --config "${selector_config}" --data-root "${data_root}" \
      --bank-root "${bank_root}" --checkpoint "${retriever_root}/best.pt" \
      --maestro-root "${SEER_MAESTRO_ROOT}" \
      --maestro-checkpoint "${maestro_checkpoint}" \
      --maestro-config "${maestro_config}" \
      --emotion2vec-model "${SEER_EMOTION2VEC_MODEL}" \
      --enhanced-root "${enhanced_root}" \
      --output "${evaluation_root}/ser" --reward-device "${device}" \
      "${source_options[@]}"
    ;;
  evaluate-wer)
    "${python_bin}" -m seer.evaluate_wer \
      --config "${selector_config}" --data-root "${data_root}" \
      --enhanced-root "${enhanced_root}" \
      --whisper-model "${SEER_WHISPER_MODEL:?Set SEER_WHISPER_MODEL in configs/paths.sh}" \
      --output "${evaluation_root}/wer" --device "${device}" \
      "${source_options[@]}"
    ;;
  *)
    echo "Usage: bash run.sh {ravdess|esd} {preprocess|build-bank|train-seer|enhance|evaluate-ser|evaluate-wer}" >&2
    exit 2
    ;;
esac
