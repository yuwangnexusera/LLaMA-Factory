# 模型加载
from llamafactory.chat import ChatModel
from llamafactory.extras.misc import torch_gc

import pandas


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
    for answer_i in answer_json:
        if num > 10:
            break
        messages = []
        messages.append({"role": "user", "content": answer_i.get("instruction") + answer_i.get("input")})
        response = ""
        response = chat_model.chat(messages)
        response = response[0].response_text
        print("input:", answer_i.get("input"))
        print("answer:", answer_i.get("output"))
        print("model answer：", response)
        num += 1
    # for new_text in chat_model.chat(messages):
    #     print(new_text, end="")
    #     response += new_text
