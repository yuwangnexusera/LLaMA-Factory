import os
# 配置 hf镜像
os.system("pip install -U huggingface_hub hf_transfer")
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 设置保存的路径
local_dir = "/mnt/windows/Users/Admin/LLM/models"

# 设置仓库id
model_id = "shenzhi-wang/Gemma-2-9B-Chinese-Chat"

cmd = f"huggingface-cli download --resume-download {model_id} --local-dir {local_dir}/{model_id} --local-dir-use-symlinks False"

# 启动下载
os.system(cmd)
