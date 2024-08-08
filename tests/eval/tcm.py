# 模型加载
from llamafactory.chat import ChatModel
from llamafactory.extras.misc import torch_gc

import json
import pandas
from datetime import datetime
from parse_ds.zy_gt import parse_json

# 20B=200亿token 0.01亿字（红楼梦）
if __name__ == "__main__":
    print("*****************运行评估测试************************")
    records = parse_json.read_json("data/match_zy/test_gt_1.json")
    args = dict(
        do_sample=True,
        model_name_or_path=" /mnt/windows/Users/Admin/LLM/models/Shanghai_AI_Laboratory/internlm2_5-7b-chat",
        adapter_name_or_path="/mnt/windows/Users/Admin/LLM/models/Shanghai_AI_Laboratory/TCM/tcm_test_800",
        template="intern2",  # 和训练保持一致
        finetuning_type="lora",  # 和训练保持一致
        # quantization_bit=4,                    # 加载 4 比特量化模型
        temperature=0.3,
        # top_p=0.7,
        max_new_tokens=256,
        repetition_penalty=1.0,
        length_penalty=1.1,
        num_beams=1,
        top_k=80,
    )
    torch_gc()
    chat_model = ChatModel(args)

    res = []
    print(f"*****************开始{datetime.now()}************************")
    for index,record in enumerate(records):
        messages = []
        messages.append({"role": "user", "content": record["question"]})
        response = ""
        response = chat_model.chat(messages)
        response = response[0].response_text
        try:
            model_answer = json.loads(response)
        except Exception as e:
            print(e)
            model_answer = response
        print(f"*****************第{index}个完成************************")
        record["model_output"] = model_answer

    print(f"*****************结束{datetime.now()}************************")
    with open("test_internlm.json", "w") as f:
        json.dump(records, f, indent=4, ensure_ascii=False)
    # for new_text in chat_model.chat(messages):
    #     print(new_text, end="")
    #     response += new_text
