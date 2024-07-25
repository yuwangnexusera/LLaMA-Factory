# 从modelscope下载

from modelscope import snapshot_download
model_dir = snapshot_download(
    "LLM-Research/Meta-Llama-3.1-8B-Instruct",
    cache_dir="/mnt/windows/Users/Admin/LLM/models/",
    ignore_file_pattern="consolidated.safetensors",
)
