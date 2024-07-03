# 从modelscope下载

from modelscope import snapshot_download
model_dir = snapshot_download("FlagAlpha/Llama3-Chinese-8B-Instruct", cache_dir="/mnt/windows/Users/Admin/LLM/models")