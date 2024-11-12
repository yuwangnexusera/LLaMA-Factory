from datetime import datetime
from llamafactory.chat import ChatModel
from llamafactory.extras.misc import torch_gc
from llamafactory.f1.recall_precise_rate import F1score
import json
from data_augmentation import sft_prompt
import pandas as pd

print("********************运行评估测试************************")
# internlm模型更好
args = dict(
    do_sample=True,
    model_name_or_path="..wwyuuuu/Qwen2___5-SFT",
    adapter_name_or_path="../DPO/wwyuuuu/dpo",  # 加载之前保存的 LoRA 适配器
    template="qwen",  # 和训练保持一致
    finetuning_type="lora",  # 和训练保持一致
    temperature=0.3,
    top_p=0.7,
    max_new_tokens=1000,
    repetition_penalty=1.2,
)

if __name__ == "__main__":
    torch_gc()
    chat_model = ChatModel(args)

    results = pd.read_json("/DPO/LLaMA-Factory/data/dpo/test_data_sft.json").to_dict(
        orient="records"
    )

    output_file_path = "/DPO/LLaMA-Factory/data/dpo/test_data_dpo.json"

    # 初始化或加载已有的结果
    try:
        with open(output_file_path, 'r', encoding='utf-8') as file:
            saved_results = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        saved_results = []

    i = 0
    for item in results:
        i += 1
        print(i)
        # 正确包装成消息列表
        instruction = f"""你的任务是根据医学实体字典：{json.dumps(item['units'],ensure_ascii=False)}，生成一份‘肺癌’的‘{item['report_type']}’.\
                            从医学实体到{item['report_type']}的生成规则如下：\
                                1、医学实体字典中的键是医学实体类型，值是该实体类型对应的具体值；\
                                2、生成的材料中必须包含医学实体中所有的值\
                                3、值为NA或者为空的键值忽略"""
        messages_2 = [{"role": "user", "content": instruction}]
        response = ""

        # 生成模型输出
        for new_text in chat_model.stream_chat(messages_2):
            print(new_text, end="")
            response += new_text
        print()

        # 将生成的结果存入新字段 sft_dpo
        item["dpo_ocr"] = response
        saved_results.append(item)

        # 实时保存每次迭代后的结果到 JSON 文件
        with open(output_file_path, 'w', encoding='utf-8') as file:
            json.dump(saved_results, file, ensure_ascii=False, indent=4)

    print("运行结束，结果已实时保存。")
