# 从modelscope下载

from modelscope import snapshot_download
model_dir = snapshot_download(
    "stepfun-ai/GOT-OCR2_0",
    cache_dir="/mnt/windows/Users/Admin/LLM/models/ocr/",
    ignore_file_pattern="consolidated.safetensors",
)
