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
    model_name_or_path="/mnt/windows/Users/Admin/LLM/models/Shanghai_AI_Laboratory/internlm2_5-7b-chat",
    adapter_name_or_path="/mnt/windows/Users/Admin/LLM/models/Shanghai_AI_Laboratory/susu_internlm2_5_v1/",  # 加载之前保存的 LoRA 适配器
    template="intern2",  # 和训练保持一致
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
    ori_data = pd.read_excel("/mnt/windows/wy/data/ss/test_data_ori.xlsx")
    report_type_units = {
            "出入院记录": 
            "门诊病历"
            "检查记录"
            "基因检测"
            "病理报告"
            "病理会诊记录"
            "其他疾病诊断书"
            "出入院疾病诊断书"
            "门诊疾病诊断书"
            "疾病诊断书"
            "病理会诊"
        }
    for index,row in ori_data.iterrows():
        url = row["url"]
        ocr_res = row["ocr_res"]
        # 分类
        category = sft_prompt.report_type_prompt.replace("{report}",ocr_res)
        if category not in report_type_units:
            continue
        prompt_1 = sft_prompt.gene_prompt_step1.replace("{category}", category).replace("{url}", url)
        print(prompt_1)
        messages = []
        messages.append({"role": "user", "content": prompt_1})
        response_1 = ""
        response_1 = chat_model.chat(messages)
        response_1 = response_1[0].response_text
        print(response_1)
        # 第二步prompt
        messages_2 = []
        prompt_2 = sft_prompt.gene_prompt_step2.replace("{list_in_values}", response_1)

        messages_2.append({"role": "user", "content": prompt_2})
        response_2 = ""
        response_2 = chat_model.chat(messages_2)
        response_2 = response_2[0].response_text
        print(response_2)
