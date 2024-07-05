# 从modelscope下载

from modelscope import snapshot_download
model_dir = snapshot_download("mixtral-8x7b-instruct-v0.1.Q3_K_M.gguf/mixtral-8x7b-instruct-v0.1.Q3_K_M.gguf", cache_dir="/mnt/windows/Users/Admin/LLM/models")