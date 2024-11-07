# 模型加载
from datetime import datetime
from llamafactory.chat import ChatModel
from llamafactory.extras.misc import torch_gc
from llamafactory.f1.recall_precise_rate import F1score
import json
from data_augmentation import sft_prompt
import pandas as pd
# 20B=200亿token 0.01亿字（红楼梦）

print("*****************运行评估测试************************")
# internlm模型更好
args = dict(
    do_sample=True,
    model_name_or_path="models/Qwen/Qwen2___5-7B-Instruct",
    adapter_name_or_path="models/Qwen/reward",  # 加载之前保存的 LoRA 适配器
    template="qwen",  # 和训练保持一致
    finetuning_type="lora",  # 和训练保持一致
    # quantization_bit=4,
    temperature=0.3,
    top_p=0.7,
    max_new_tokens=1024,
    repetition_penalty=1.0,
    length_penalty=1.1,
)
if __name__ == "__main__":
    torch_gc()
    chat_model = ChatModel(args)
    ret_data = []

    messages = []
    messages.append({"role": "user", "content": "哈哈哈"})
    response_1 = ""
    response_1 = chat_model.chat(messages)
    response_1 = response_1[0].response_text
    print(response_1)
