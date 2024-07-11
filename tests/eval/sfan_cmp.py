# 模型加载
from llamafactory.chat import ChatModel
from llamafactory.extras.misc import torch_gc

import json
import pandas
from datetime import datetime

def compare_annotations(correct_annotations, model_annotations):
    # 按照匹配标准排序两个列表
    correct_annotations_sorted = sorted(
        correct_annotations, key=lambda x: (x["entity_type"], x["entity_value"], x["start_position"], x["end_position"])
    )
    model_annotations_sorted = sorted(
        model_annotations, key=lambda x: (x["entity_type"], x["entity_value"], x["start_position"], x["end_position"])
    )

    mismatches = []

    # 比较排序后的列表
    for correct_ann, model_ann in zip(correct_annotations_sorted, model_annotations_sorted):
        if (
            correct_ann["entity_type"] != model_ann["entity_type"]
            or correct_ann["entity_value"] != model_ann["entity_value"]
            or correct_ann["start_position"] != model_ann["start_position"]
            or correct_ann["end_position"] != model_ann["end_position"]
        ):
            mismatches.append((correct_ann, model_ann))

    return mismatches


# 20B=200亿token 0.01亿字（红楼梦）
if __name__ == "__main__":
    print("*****************运行评估测试************************")
    answer_json = pandas.read_json("data/Sfan/ner_sfan_test.json", orient="records").to_dict(orient="records")
    args = dict(
        do_sample=True,
        model_name_or_path="/mnt/windows/Users/Admin/LLM/models/AI-ModelScope/gemma-2b-it",
        adapter_name_or_path="/mnt/windows/Users/Admin/LLM/models/AI-ModelScope/test/sfan_gemma2b_test_v1",
        template="gemma",  # 和训练保持一致
        finetuning_type="lora",  # 和训练保持一致
        # quantization_bit=4,                    # 加载 4 比特量化模型
        temperature=0.3,
        top_p=0.7,
        max_new_tokens=512,
        repetition_penalty=1.0,
        length_penalty=1.1,
        num_beams=3,
        top_k=80,
    )
    torch_gc()
    chat_model = ChatModel(args)
    num = 1
    res = []
    print(f"*****************开始{datetime.now()}************************")
    for answer_i in answer_json:
        if num > 100:
            break
        messages = []
        messages.append({"role": "user", "content": answer_i.get("instruction") + answer_i.get("input")})
        response = ""
        response = chat_model.chat(messages)
        response = response[0].response_text
        try:
            model_answer = json.loads(response)
        except Exception as e:
            print(e)
            model_answer = response
        print("*****************第" + str(num) + "个完成************************")
        res.append(
            {
                "input": answer_i.get("input"),
                "answer": json.loads(answer_i.get("output")),
                "model answer": model_answer,
            }
        )
        num += 1
    print(f"*****************结束{datetime.now()}************************")
    with open("../res.json", "w") as f:
        json.dump(res, f, indent=4, ensure_ascii=False)
    # for new_text in chat_model.chat(messages):
    #     print(new_text, end="")
    #     response += new_text
