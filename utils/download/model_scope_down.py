# 从modelscope下载

from modelscope import snapshot_download
model_dir = snapshot_download(
    "wwyuuuu/Qwen2.5-SFT",
    cache_dir="../wwyuuuu/",
    ignore_file_pattern="consolidated.safetensors",
)
