# 从modelscope下载

from modelscope import snapshot_download
model_dir = snapshot_download(
    "AI-ModelScope/gemma-2b-it", cache_dir="/mnt/windows/Users/Admin/LLM/models/", ignore_file_pattern="*.gguf"
)
