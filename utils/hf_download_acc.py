import os
# 配置 hf镜像
os.system("pip install -U huggingface_hub hf_transfer")
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 设置保存的路径
local_dir = "/mnt/windows/Users/Admin/LLM/models"

# 设置仓库id
model_id = "n810x/Mixtral-8x7B-Instruct-v0.1-3bit-exl2"

exclude = "*.gguf"
cmd = f"huggingface-cli download --resume-download {model_id} --local-dir {local_dir}/{model_id} --local-dir-use-symlinks False --exclude {exclude}"

# 启动下载
os.system(cmd)
