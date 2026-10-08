# Copy this file to configs/paths.sh and replace every /absolute/path entry.

# Raw corpora downloaded from their official sources. This can be a mounted
# data directory linked from data/raw/ in the checkout.
export SEER_DATA_ROOT="/absolute/path/to/seer-data"
export SEER_RAVDESS_RAW="${SEER_DATA_ROOT}/raw/ravdess"
export SEER_ESD_RAW="${SEER_DATA_ROOT}/raw/esd"

# External repositories and pretrained models. These are not distributed by SEER.
export SEER_MAESTRO_ROOT="/absolute/path/to/MaestroEVC"
export SEER_EMOTION_DIARIZATION_MODEL="/absolute/path/to/emotion-diarization-wavlm-large"
export SEER_SPEAKER_MODEL="/absolute/path/to/spkrec-ecapa-voxceleb"
export SEER_EMOTION2VEC_MODEL="/absolute/path/to/emotion2vec_plus_large"
# Required only for the main-table WER evaluation.
export SEER_WHISPER_MODEL="/absolute/path/to/whisper-large-v3-turbo"

# Features, model caches, checkpoints, and generated audio.
export SEER_RUN_ROOT="/absolute/path/to/seer-runs"
export SEER_FEATURE_CACHE="${SEER_RUN_ROOT}/model-cache"

export SEER_RAVDESS_DATA="${SEER_RUN_ROOT}/ravdess/data"
export SEER_RAVDESS_MAESTRO_DIR="${SEER_RUN_ROOT}/ravdess/maestro"
export SEER_RAVDESS_MAESTRO_CHECKPOINT="${SEER_RAVDESS_MAESTRO_DIR}/g_00023040"
export SEER_RAVDESS_MAESTRO_CONFIG="${SEER_RAVDESS_MAESTRO_DIR}/config.json"
export SEER_RAVDESS_BANK="${SEER_RUN_ROOT}/ravdess/reference-bank"
export SEER_RAVDESS_RETRIEVER="${SEER_RUN_ROOT}/ravdess/retriever"
export SEER_RAVDESS_OUTPUT="${SEER_RUN_ROOT}/ravdess/enhanced"
export SEER_RAVDESS_EVALUATION="${SEER_RUN_ROOT}/ravdess/evaluation"

export SEER_ESD_DATA="${SEER_RUN_ROOT}/esd-en/data"
export SEER_ESD_MAESTRO_DIR="${SEER_RUN_ROOT}/esd-en/maestro"
export SEER_ESD_MAESTRO_CHECKPOINT="${SEER_ESD_MAESTRO_DIR}/g_00150000"
export SEER_ESD_MAESTRO_CONFIG="${SEER_ESD_MAESTRO_DIR}/config.json"
export SEER_ESD_BANK="${SEER_RUN_ROOT}/esd-en/reference-bank"
export SEER_ESD_RETRIEVER="${SEER_RUN_ROOT}/esd-en/retriever"
export SEER_ESD_OUTPUT="${SEER_RUN_ROOT}/esd-en/enhanced"
export SEER_ESD_EVALUATION="${SEER_RUN_ROOT}/esd-en/evaluation"

# Optional command/device overrides.
export SEER_PYTHON="python"
export SEER_DEVICE="cuda"
