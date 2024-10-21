import os
# 配置 hf镜像
os.system("pip install -U huggingface_hub hf_transfer")
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 设置保存的路径
local_dir = "/mnt/windows/Users/Admin/LLM/models/"

# 设置仓库id
model_id = "google-bert/bert-base-chinese"

exclude = "*.pth" #--exclude {exclude}
cmd = f"huggingface-cli download --resume-download {model_id} --local-dir {local_dir}/{model_id} --exclude {exclude} --local-dir-use-symlinks False --token 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"

# 启动下载
os.system(cmd)
