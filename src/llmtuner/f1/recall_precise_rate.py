import json
from copy import deepcopy
from difflib import SequenceMatcher
import re
import numpy as np
from typing import List,Tuple
from pypinyin import pinyin, Style

class F1score:

    def compare_json(self, answer, generate):
        # 此函数需要处理不同数据类型的 JSON 对象，比较它们的内容
        if isinstance(answer, dict) and isinstance(generate, dict):
            keys = set(answer.keys()) | set(generate.keys())
            score = 0
            total_keys = len(keys)
            for key in keys:
                if key in answer and key in generate:
                    sub_score = self.compare_json(answer[key], generate[key])
                    if sub_score is not None:
                        score += sub_score
            return score / total_keys if total_keys > 0 else 1
        elif isinstance(answer, list) and isinstance(generate, list):
            min_len = min(len(answer), len(generate))
            max_len = max(len(answer), len(generate))
            score = 0
            for i in range(min_len):
                sub_score = self.compare_json(answer[i], generate[i])
                if sub_score is not None:
                    score += sub_score
            # 对于列表长度不等的情况，超出的部分得分为 0
            return score / max_len if max_len > 0 else 1
        else:
            return 1 if answer == generate else 0

    def sort_similarity_matrix(self, answer, generate) -> List[Tuple[float, Tuple[int, int]]]:
        len_answer = len(answer)
        len_generate = len(generate)
        sim_matrix = -np.ones((len_answer, len_generate))
        for i in range(len_answer):
            for j in range(len_generate):
                sim_matrix[i][j] = self.compare_json(answer[i], generate[j])
        indexed_scores = []
        for i in range(sim_matrix.shape[0]):
            for j in range(sim_matrix.shape[1]):
                indexed_scores.append((sim_matrix[i, j], (i, j)))

        # 按相似度排序
        indexed_scores.sort(reverse=True, key=lambda x: x[0])
        return indexed_scores

    # 用于实验过程中不区分单元的比较
    def labor_recall_precise(self, generated_answer, answer_json, include_na_in_total=False):
        """
        CE (Correct Extraction) ↔ TP (True Positive)
        IE (Incorrect Extraction) + SE (Spurious Extraction) ↔ FP (False Positive)
        ME (Missed Extraction) ↔ FN (False Negative)
        TN (True Negative): 在信息提取任务中无法定义。
        """
        unit_loc_mapping = {
            "Basic Information": ["Date of Birth", "Age", "Gender"],
            "Disease": [
                "Date of First Diagnosis",
                "Time of First Pathological Diagnosis (Biopsy, Post-operative Pathology, etc.)",
                "Time of First Lung Resection",
                "Time of First Imaging Diagnosis",
                "Time of First Treatment (Drugs, Radiotherapy, etc.)",
                "Time of First Symptom",
                "Disease Name",
            ],
            "Symptom": ["ECOG Score", "ECOG Date"],
            "Diagnosis": ["Diagnosing Doctor"],
            "Imaging": ["Brain Metastasis Date", "Brain Metastasis Site"],
            "Pathology": ["Pathology Date", "Pathology Type"],
            "Genetic Testing": [
                "ALK",
                "MET",
                "RB1",
                "RET",
                "BRAF",
                "BRCA",
                "EGFR",
                "FGFR",
                "KRAS",
                "NTRK",
                "ROS1",
                "TP53",
                "KEAP1",
                "STK11",
                "HER2 (ERBB2)",
                "HER3 (ERBB3)",
                "HER4 (ERBB4)",
                "Genetic Testing Date",
            ],
            "Immune Testing": [
                "Immune Cell",
                "Combined Positive Score",
                "Tumor Proportion Score",
                "PD-L1",
                "Immunological Test Date",
            ],
            "Cancer treatment": [
                "Surgical Site",
                "Treatment Start Date",
                "Treatment Drug Names",
                "Treatment End Date",
                "Specific Tumor Treatment Method",
            ],
            "Treatment Drug Plan": [
                "Treatment Start Date",
                "Treatment Drug Names",
                "Treatment End Date",
                "Is Treatment Drug Recommended",
            ],
            "Comorbid Disease": [
                "Date of Confirmed Disease",
                "Information Source",
                "Infectious Diseases",
                "Respiratory System Diseases",
                "Circulatory System Diseases",
                "Malignant Tumor Conditions",
                "Digestive System Diseases",
                "Nervous System Diseases",
                "Urogenital System Diseases",
                "Eye, Ear, Nose, and Throat Related Diseases",
                "Endocrine and Immune System Diseases",
            ],
            "Date": ["Admission Date", "Discharge Date", "Medical History Collection Date", "Record Date"],
        }

        # 单元层级
        ce = ie = me = se = 0  # 初始化计数器：正确提取、错误提取、漏提取、误提取
        error_keys = []
        try:
            for unit_name, answer_unit_value in answer_json.items():
                generate_unit_value = generated_answer.get(unit_name)
                if generate_unit_value is None:
                    me += len(answer_unit_value) * len(unit_loc_mapping[unit_name])
                    continue
                length_gap = len(answer_unit_value) - len(generate_unit_value)
                if length_gap > 0: # 生成的数据更少
                    me += length_gap * len(unit_loc_mapping[unit_name])
                elif length_gap < 0: # 生成的数据更多
                    se += abs(length_gap) * len(unit_loc_mapping[unit_name])
                similarity_tuple_list = self.sort_similarity_matrix(answer_unit_value, generate_unit_value) # (相似度, (i, j))
                compared_generate_index = set()
                compared_answer_index = set()
                for similarity, (answer_index, generate_index) in similarity_tuple_list:
                    if generate_index in compared_generate_index or answer_index in compared_answer_index: # 重复的数据
                        continue
                    compared_generate_index.add(generate_index)
                    compared_answer_index.add(answer_index)
                    # 取出对应的两个value 字典做对比
                    answer_unit_value_dict = answer_unit_value[answer_index]
                    generate_unit_value_dict = generate_unit_value[generate_index]
                    for k_a, v_a in answer_unit_value_dict.items():
                        # 数据准备
                        # 先全部转换为小写
                        # 可能出现整型数字，也可能是整型字符串，做一次归一化，变成字符串
                        # 空串空列表都映射为na
                        #诊断医生转为拼音
                        # 数据准备结束
                        # 列表+长度为1，取出这个值；列表长度不是1，转为set，比较列表。
                        if k_a not in generate_unit_value_dict: #生成数据缺失key
                            me += 1
                            error_keys.append({k_a: "key not exist"})
                            continue
                        v_g = generate_unit_value_dict[k_a]
                        if isinstance(v_a, list) and isinstance(v_g, list):
                            if set(v_a) != set(v_g):
                                ie += 1
                                error_keys.append({k_a: {"answer": v_a, "generate": v_g}})
                        elif v_a != v_g:
                            ie += 1
                            error_keys.append({k_a: {"answer": v_a, "generate": v_g}})
                        elif v_a!="na" and v_g=="na":
                            me+=1
                            error_keys.append({k_a: {"answer": v_a, "generate": v_g}})
                        else:
                            ce+= 1
                #  按照列表之前json对象的相似性，计算
                # 先全部转换为小写
                for k_g, v_g in generate_unit_value.items():
                    if isinstance(v_g, list):
                        generate_unit_value[k_g] = [i.lower() for i in v_g]
                    else:
                        if isinstance(v_g, str):
                            generate_unit_value[k_g] = v_g.lower()
                for k_a, v_a in answer_unit_value.items():
                    if isinstance(v_a, list):
                        answer_unit_value[k_a] = [i.lower() for i in v_a]
                    else:
                        if isinstance(v_a, str):
                            answer_unit_value[k_a] = v_a.lower()
                # 从 answer_json 获取所有可能的键作为参考
                reference_keys = set(answer_unit_value[0].keys() if isinstance(answer_unit_value, list) else answer_unit_value.keys())
                generated_keys = set(generate_unit_value[0].keys() if isinstance(generate_unit_value, list) else generate_unit_value.keys())

                # 计算 CE 和 IE TODO 需优化
                for key in reference_keys:
                    # 准备操作 start
                    # 可能出现整型数字，也可能是整型字符串，做一次归一化，变成字符串
                    generate_unit_value[key] = (
                        str(generate_unit_value[key])
                        if isinstance(generate_unit_value[key], int)
                        else generate_unit_value[key]
                    )
                    answer_unit_value[key] = (
                        str(answer_unit_value[key]) if isinstance(generate_unit_value[key], int) else answer_unit_value[key]
                    )
                    # 空串空列表都映射为na
                    if generate_unit_value[key] == "" or generate_unit_value[key] == []:
                        generate_unit_value[key] = "na"
                    if answer_unit_value[key] == "" or answer_unit_value[key] == []:
                        answer_unit_value[key] = "na"
                    if key in ["Diagnosing Doctor"]: #诊断医生转为拼音
                        pinyin_list_a = pinyin(answer_unit_value[key], style=Style.NORMAL)
                        pinyin_list_g = pinyin(generate_unit_value[key], style=Style.NORMAL)
                        answer_unit_value[key] = "".join(word[0] for word in pinyin_list_a)
                        generate_unit_value[key] = "".join(word[0] for word in pinyin_list_g)
                    # 准备操作 end
                    if key in generated_keys or (generate_unit_value[key] == "na" and answer_unit_value[key] != "na"):
                        # 数据准备，列表+长度为1，取出这个值；列表长度不是1，转为set，比较列表。
                        if isinstance(generate_unit_value[key], list):
                            if len(generate_unit_value[key]) == 1:
                                generate_unit_value[key] = generate_unit_value[key][0]
                            else:
                                generate_unit_value[key] = set(generate_unit_value[key])
                        if isinstance(answer_unit_value[key], list):
                            if len(answer_unit_value[key]) == 1:
                                answer_unit_value[key] = answer_unit_value[key][0]
                            else:
                                answer_unit_value[key] = set(answer_unit_value[key])
                        if key in [
                            "PD-L1",
                            "Age",
                            "Immune Cell",
                            "Tumor Proportion Score",
                            "Combined Positive Score",
                        ] and isinstance(
                            generate_unit_value[key], str
                        ):  # 只比较数字部分
                            generate_unit_value[key] = (
                                re.findall(r"\d+", generate_unit_value[key])[0]
                                if re.findall(r"\d+", generate_unit_value[key])
                                else "na"
                            )
                            answer_unit_value[key] = (
                                re.findall(r"\d+", answer_unit_value[key])[0]
                                if re.findall(r"\d+", answer_unit_value[key])
                                else "na"
                            )

                        if generate_unit_value[key] == answer_unit_value[key]:
                            ce += 1  # 提取正确
                        else:
                            ie += 1  # 提取错误
                            error_keys.append({key: [answer_unit_value[key], generate_unit_value[key]]})
                    else:
                        me += 1  # 漏提取

                    # 计算 SE
                    for key in generated_keys:
                        if key not in reference_keys:
                            se += 1  # 误提取

                    # 计算 Precision 和 Recall
                    precision = ce / (ce + ie + se) if (ce + ie) > 0 else 0
                    recall = ce / (ce + me) if (ce + me) > 0 else 0

            return {
                "correct_extraction": ce,
                "incorrect_extraction": ie,
                "missed_extraction": me,
                "spurious_extraction": se,
                "precision": precision,
                "recall": recall,
                "error_keys": error_keys,
            }
        except Exception as e:
            return {
                "correct_extraction": 0,
                "incorrect_extraction": 0,
                "missed_extraction": 0,
                "spurious_extraction": 0,
                "precision": 0,
                "recall": 0,
                "error_keys": str(e),
            }


if __name__ == "__main__":
    f1 = F1score()
    # 示例数据
    generated_answer = [
        {
            "Date of First Diagnosis": "2021-12-12",
            "Time of First Pathological Diagnosis (Biopsy, Post-operative Pathology, etc.)": "na",
            "Time of First Lung Resection": "na",
            "Time of First Imaging Diagnosis": "2020-07-33",
            "Time of First Treatment (Drugs, Radiotherapy, etc.)": "na",
            "Time of First Symptom": "na",
            "Disease Name": ["Mesothelioma", "Small cell carcinoma"],
        }
    ]
    # 其他单元
    answer_json = [
        {
            "Date of First Diagnosis": "2021-12-17",
            "Time of First Pathological Diagnosis (Biopsy, Post-operative Pathology, etc.)": "na",
            "Time of First Lung Resection": "na",
            "Time of First Imaging Diagnosis": "2020-07-31",
            "Time of First Treatment (Drugs, Radiotherapy, etc.)": "na",
            "Time of First Symptom": "na",
            "Disease Name": ["Mesothelioma", "Small cell carcinoma"],
        },
        {
            "Date of First Diagnosis": "2021-12-12",
            "Time of First Pathological Diagnosis (Biopsy, Post-operative Pathology, etc.)": "na",
            "Time of First Lung Resection": "na",
            "Time of First Imaging Diagnosis": "2020-07-33",
            "Time of First Treatment (Drugs, Radiotherapy, etc.)": "na",
            "Time of First Symptom": "na",
            "Disease Name": ["Mesothelioma", "Small cell carcinoma"],
        },
    ]

    # 调用函数并打印结果
    result = f1.reorder_json_lists(answer_json, generated_answer)
    print(result)
