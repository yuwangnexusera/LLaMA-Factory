# 从modelscope下载

from modelscope import snapshot_download
model_dir = snapshot_download("qwen/Qwen1.5-14B-Chat-GPTQ-Int8", cache_dir="/mnt/windows/Users/Admin/LLM/models")