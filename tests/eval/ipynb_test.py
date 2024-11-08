# 模型加载
from datetime import datetime
from llamafactory.chat import ChatModel
from llamafactory.extras.misc import torch_gc
from llamafactory.f1.recall_precise_rate import F1score
import json
from data_augmentation.sft_prompt import sft_unit_prompt
# 20B=200亿token 0.01亿字（红楼梦）
print("*****************运行评估测试************************")
# internlm模型更好
args = dict(
    do_sample=True,
    model_name_or_path="/mnt/windows/Users/Admin/LLM/models/qwen/Qwen2-7B",
    adapter_name_or_path="/mnt/windows/Users/Admin/LLM/models/qwen/test_7B/base_qwen2",  # 加载之前保存的 LoRA 适配器
    template="qwen",  # 和训练保持一致
    finetuning_type="lora",  # 和训练保持一致
    # quantization_bit=4,
    temperature=0.3,
    top_p=0.7,
    max_new_tokens=1024,
    repetition_penalty=1.0,
    length_penalty=1.1,
    num_beams=1,
    top_k=80,
)
torch_gc()
chat_model = ChatModel(args)
f1_cal = F1score()
# 测试推理

unit_name = "治疗用药方案"
test_path_mapping = {
    "病理": "data/Pathology/test_zh.json",
    "治疗用药方案": "data/Treatment Drug Plan/test_zh.json",
    "基本信息": "data/Basic Information/test_zh.json",
    "疾病": "data/Disease/test_zh.json",
    "体征数据": "data/Symptom/test_zh.json",
    "诊断": "data/Diagnosis/test_zh.json",
    "影像学": "data/Imaging/test_zh.json",
    "基因检测": "data/Genetic Testing/test_zh.json",
    "免疫检测": "data/Immune Testing/test_zh.json",
    "肿瘤治疗": "data/Cancer treatment/test_zh.json",
    "合并疾病": "data/Comorbid Disease/test_zh.json",
    "日期": "data/date_unit/test_zh.json",
}


def generate_prompt_extract(content):
    return sft_unit_prompt.get(unit_name) + content


with open(test_path_mapping.get(unit_name), "r", encoding="utf-8") as file:
    data = json.load(file)
    not_json_num = 0
    res_eval_metrics = {
        "correct_extraction": 0,
        "incorrect_extraction": 0,
        "missed_extraction": 0,
        "spurious_extraction": 0,
        "precision": 0,
        "recall": 0,
    }
    res_cmp = []
    for index, report in enumerate(data):
        messages = []
        content = report.get("input", "")
        if content:
            query = generate_prompt_extract(content)
            messages.append({"role": "user", "content": query})
            response = ""
            print(f"{index}推理开始")
            response = chat_model.chat(messages)
            response = response[0].response_text
            print(response)
            # for new_text in chat_model.chat(messages):
            #     print(new_text, end="")
            #     response += new_text
            print(f"答案：{report.get('output')}")
            try:
                if "```json" in response:
                    response = response.replace("```json", "").replace("```", "")
                generated_answer = json.loads(response)
            except Exception as e:
                print("生成结果非json")
                not_json_num += 1
                generated_answer = {}
                continue
            eval_metrics = f1_cal.labor_recall_precise(
                {unit_name: generated_answer}, {unit_name: json.loads(report.get("output"))}
            )
            res_cmp.append(
                {"report": content, "answer": report.get("output"), "response": generated_answer, "eval": eval_metrics}
            )
            print(eval_metrics)
            res_eval_metrics["correct_extraction"] += eval_metrics.get("correct_extraction", 0)
            res_eval_metrics["incorrect_extraction"] += eval_metrics.get("incorrect_extraction", 0)
            res_eval_metrics["missed_extraction"] += eval_metrics.get("missed_extraction", 0)
            res_eval_metrics["spurious_extraction"] += eval_metrics.get("spurious_extraction", 0)
            res_eval_metrics["precision"] += eval_metrics.get("precision", 0)
            res_eval_metrics["recall"] += eval_metrics.get("recall", 0)
    res_eval_metrics["precision"] = res_eval_metrics["precision"] / len(data)
    res_eval_metrics["recall"] = res_eval_metrics["recall"] / len(data)
    P = res_eval_metrics["precision"]
    R = res_eval_metrics["recall"]
    print(f"评估结果：{res_eval_metrics},F1:{2*P*R/(P+R)}")
    # with open(f"../res_eval_{unit_name}{datetime.now().strftime('%Y%m%d%H%M%S')}.txt", "w", encoding="utf-8") as file:
    #     file.write("评估结果：\n")
    #     for key, value in res_eval_metrics.items():
    #         file.write(f"{key}: {value}\n")

    #     # 计算并写入 F1 值
    #     F1 = 2 * P * R / (P + R)
    #     file.write(f"F1: {F1}\n")
    # with open(f"../res_cmp_{unit_name}{datetime.now().strftime('%Y%m%d%H%M%S')}.json", "w", encoding="utf-8") as file:
    #     json.dump(res_cmp, file, indent=4, ensure_ascii=False)
