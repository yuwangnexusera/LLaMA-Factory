from transformers import AutoTokenizer, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("/mnt/windows/Users/Admin/LLM/models/qwen/Qwen2___5-7B-Instruct")
print(model)
