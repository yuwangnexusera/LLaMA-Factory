from modelscope.hub.api import HubApi

YOUR_ACCESS_TOKEN = "0976a8ad-3159-4256-8944-e8271f82a882"

api = HubApi()
api.login(YOUR_ACCESS_TOKEN)

# 上传模型
api.push_model(
    model_id="wwyuuuu/Qwen2.5-SFT",
    model_dir="/mnt/windows/Users/Admin/LLM/models/qwen/qwen-rlhf/sft",
)
