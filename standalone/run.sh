export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

uv run uvicorn standalone.main:app --reload
