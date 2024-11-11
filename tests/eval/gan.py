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
    model_name_or_path="/mnt/windows/Users/Admin/LLM/models/qwen/qwen-rlhf/sft",
    # adapter_name_or_path="/mnt/windows/Users/Admin/LLM/models/qwen/gan/Qwen2_5-7B-gan",  # 加载之前保存的 LoRA 适配器
    template="qwen",  # 和训练保持一致
    finetuning_type="lora",  # 和训练保持一致
    # quantization_bit=4,
    temperature=0.95,
    top_p=0.7,
    max_new_tokens=1000,
    repetition_penalty=1.2,
)
if __name__ == "__main__":
    torch_gc()
    chat_model = ChatModel(args)
    results = []
    path = "/root/LLM/LLaMA-Factory/data/gan/test_data_sft.json"
    output_file_path = "/root/LLM/LLaMA-Factory/data/gan/dpo.json"
    ori_data = pd.read_json(path).to_dict(orient="records")
    i = 0
    for item in ori_data:
        i += 1
        print(i)
        # 正确包装成 messages 列表
        instruction = f"你的任务是根据给定的医学实体，生成类型为:{item['report_type']}的报告, 报告中必须包含医学实体中的所有值. 医学实体：{json.dumps(item['units'],ensure_ascii=False)}"

        # 将字符串 instruction 包装为消息列表，传递给 chat_model
        messages_2 = [{"role": "user", "content": instruction}]
        response = ""
        # 生成模型输出
        for new_text in chat_model.stream_chat(messages_2):
            print(new_text, end="")
            response += new_text
        print()
        # 将生成的 OCR 结果存入新字段 sft_ocr
        item["dpo_ocr"] = response

        results.append(item)
        # 实时保存每次迭代后的结果到 JSON 文件
        with open(output_file_path, "w", encoding="utf-8") as file:
            json.dump(results, file, ensure_ascii=False, indent=4)
