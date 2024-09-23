# 从modelscope下载

from modelscope import snapshot_download
model_dir = snapshot_download(
    "qwen/Qwen2.5-7B-Instruct",
    cache_dir="/mnt/windows/Users/Admin/LLM/models/",
    ignore_file_pattern="consolidated.safetensors",
)
