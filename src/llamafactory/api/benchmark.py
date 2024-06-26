from typing import List

from ..data import Role as DataRole
from ..extras.logging import get_logger
from .common import dictify, jsonify
from ..chat import ChatModel
from .protocol import Role, BenchmarkRequest, BenchmarkResponse
import json
from ..f1 import F1score


logger = get_logger(__name__)
ROLE_MAPPING = {
    Role.USER: DataRole.USER.value,
    Role.ASSISTANT: DataRole.ASSISTANT.value,
    Role.SYSTEM: DataRole.SYSTEM.value,
    Role.FUNCTION: DataRole.FUNCTION.value,
    Role.TOOL: DataRole.OBSERVATION.value,
}

from parse_ds.sft_prompt import sft_unit_prompt


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


# 测试集的操作搬过来
def ie_unit_benchmark(request: "BenchmarkRequest", chat_model: ChatModel):
    f1_cal = F1score()
    # 如果有prompt,就使用新传入的,若没有,使用默认的
    unit_names = request.test_unit
    evaluation_criteria = {}  # 评估指标
    model_correct_answer = {}  # 模型回答与正确答案
    error_details = {}
    for unit_name in unit_names:
        eval_metrics = {
            "correct_extraction": 0,
            "incorrect_extraction": 0,
            "missed_extraction": 0,
            "spurious_extraction": 0,
            "precision": 0,
            "recall": 0,
        }  # 单元的评估指标
        prompt = sft_unit_prompt.get(unit_name)
        if unit_name not in test_path_mapping:
            continue
        if unit_name not in model_correct_answer:
            model_correct_answer[unit_name] = []
        with open(test_path_mapping.get(unit_name), "r", encoding="utf-8") as file:
            data = json.load(file)
            for index, report in enumerate(data):
                if len(model_correct_answer[unit_name]) >= request.samples:
                    break
                messages = []
                content = report.get("input", "")
                messages.append({"role": "user", "content": prompt + content})
                response = ""
                for new_text in chat_model.stream_chat(
                    messages=messages,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    max_new_tokens=request.max_new_tokens,
                    repetition_penalty=request.repetition_penalty,
                    length_penalty=request.length_penalty,
                ):
                    response += new_text
                try:
                    if "```json" in response:
                        response = response.replace("```json", "").replace("```", "")
                    generated_answer = json.loads(response)
                    report_eval_metrics = f1_cal.labor_recall_precise(
                        {unit_name: generated_answer}, {unit_name: json.loads(report.get("output"))}
                    )
                except Exception as e:
                    generated_answer = response #非json直接返回

                model_correct_answer[unit_name].append(
                    {
                        f"report_{index}": content,
                        "generate": generated_answer,
                        "answer": json.loads(report.get("output")),
                    }
                )
                eval_metrics["correct_extraction"] += report_eval_metrics.get("correct_extraction", 0)
                eval_metrics["incorrect_extraction"] += report_eval_metrics.get("incorrect_extraction", 0)
                eval_metrics["missed_extraction"] += report_eval_metrics.get("missed_extraction", 0)
                eval_metrics["spurious_extraction"] += report_eval_metrics.get("spurious_extraction", 0)
                eval_metrics["precision"] += report_eval_metrics.get("precision", 0)
                eval_metrics["recall"] += report_eval_metrics.get("recall", 0)
            for key in eval_metrics:
                eval_metrics[key] = eval_metrics[key] / len(model_correct_answer[unit_name])
            evaluation_criteria[unit_name] = eval_metrics
    return BenchmarkResponse(evaluation_criteria=evaluation_criteria, model_correct_answer=model_correct_answer, error_details={"":"TODO"})
