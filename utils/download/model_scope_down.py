# 从modelscope下载

from modelscope import snapshot_download
model_dir = snapshot_download(
    "AI-ModelScope/Mistral-Nemo-Instruct-2407",
    cache_dir="/mnt/windows/Users/Admin/LLM/models/",
    ignore_file_pattern="consolidated.safetensors",
)
