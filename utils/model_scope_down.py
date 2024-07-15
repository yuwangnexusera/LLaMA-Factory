# 从modelscope下载

from modelscope import snapshot_download
model_dir = snapshot_download(
    "qwen/Qwen1.5-14B-Chat", cache_dir="/mnt/windows/Users/Admin/LLM/models/", ignore_file_pattern=".*\.gguf"
)
