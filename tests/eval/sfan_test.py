# 模型加载
from datetime import datetime
from llamafactory.chat import ChatModel
from llamafactory.extras.misc import torch_gc
from llamafactory.f1.recall_precise_rate import F1score
import json
from parse_ds.sft_prompt import sft_unit_prompt
# 20B=200亿token 0.01亿字（红楼梦）
print("*****************运行评估测试************************")
args = dict(
    do_sample=True,
    model_name_or_path="/mnt/windows/Users/Admin/LLM/models/AI-ModelScope/gemma-2b-it",
    adapter_name_or_path="/mnt/windows/Users/Admin/LLM/models/AI-ModelScope/sfan_gemma2b_test",  
    template="gemma",  # 和训练保持一致
    finetuning_type="lora",  # 和训练保持一致
    # quantization_bit=4,                    # 加载 4 比特量化模型
    temperature=0.5,
    top_p=0.7,
    max_new_tokens=512,
    repetition_penalty=1.0,
    length_penalty=1.1,
    num_beams=3,
    top_k=80,
)
torch_gc()
chat_model = ChatModel(args)

messages = []
prompt = "Your mission is to extract entity information from biomedical text predictions, including Selin, Spessis, Jean, Technik, Protan, Proses, and Chemical. biomedical text ：Mitochondrial dynamics and metabolites reciprocally influence each other. Mitochondrial-derived vesicles (MDVs) transport damaged mitochondrial components to lysosomes or the extracellular space, and they are an emerging mechanism that regulates mitochondrial quality."
messages.append({"role": "user", "content": prompt})
response = ""
response = chat_model.chat(messages)
response = response[0].response_text
print("model answer：",response)
# for new_text in chat_model.chat(messages):
#     print(new_text, end="")
#     response += new_text

