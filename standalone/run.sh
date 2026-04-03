export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

explorer.exe http://127.0.0.1:8000/ui
uv run uvicorn standalone.main:app --reload
