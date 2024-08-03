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
    reports = parse_json.read_json("data/test_gt.json")
    args = dict(
        do_sample=True,
        model_name_or_path="/mnt/windows/Users/Admin/LLM/models/qwen/TCM/tcm_qwen2_v1",
        # adapter_name_or_path="/mnt/windows/Users/Admin/LLM/models/AI-ModelScope/test/sfan_gemma2b_test_BC5CDR",
        template="qwen",  # 和训练保持一致
        # finetuning_type="lora",  # 和训练保持一致
        # quantization_bit=4,                    # 加载 4 比特量化模型
        temperature=0.1,
        top_p=0.7,
        max_new_tokens=512,
        repetition_penalty=1.1,
        length_penalty=1.1,
        num_beams=2,
        top_k=80,
    )
    torch_gc()
    chat_model = ChatModel(args)

    res = []
    print(f"*****************开始{datetime.now()}************************")
    for index_report, report in enumerate(reports):
        for index, question in enumerate(report):
            messages = []
            messages.append({"role": "user", "content": str(question[0])})
            response = ""
            response = chat_model.chat(messages)
            response = response[0].response_text
            try:
                model_answer = json.loads(response)
            except Exception as e:
                print(e)
                model_answer = response
            print(f"*****************{index_report}份报告{index}个问题{response}************************")
            res.append(
                {
                    "input": question[0],
                    "answer": question[1],
                    "model answer": model_answer,
                }
            )
    print(f"*****************结束{datetime.now()}************************")
    with open("test.json", "w") as f:
        json.dump(res, f, indent=4, ensure_ascii=False)
    # for new_text in chat_model.chat(messages):
    #     print(new_text, end="")
    #     response += new_text
