import json
from copy import deepcopy
from difflib import SequenceMatcher
import re
import numpy as np
from typing import List,Tuple
from pypinyin import pinyin, Style
from datetime import datetime
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
                "Whether to undergo genetic testing",
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
            "基本信息": ["出生日期", "年龄", "性别"],
            "疾病": [
                "疾病首次确诊日期",
                "第一次病理确诊时间（穿刺、术后病理等）",
                "第一次切肺手术时间",
                "第一次影像确诊时间",
                "第一次治疗时间（药物、放疗等）",
                "首发症状时间",
                "疾病名称",
            ],
            "体征数据": ["ECOG", "ECOG日期"],
            "诊断": ["诊断医生"],
            "影像学": ["脑转移日期", "脑转部位"],
            "病理": ["病理日期", "病理类型"],
            "基因检测": [
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
                "HER2(ERBB2)",
                "HER3（ERBB3）",
                "HER4（ERBB4）",
                "基因检测日期",
            ],
            "免疫检测": ["IC", "CPS", "TPS", "PDL1", "免疫检测日期"],
            "肿瘤治疗": ["手术部位", "治疗开始日期", "治疗用药名称", "治疗结束日期", "肿瘤具体治疗方式"],
            "治疗用药方案": ["治疗开始日期", "治疗用药名称", "治疗结束日期"],
            "合并疾病": [
                "合并疾病确诊日期",
                "信息来源",
                "传染性疾病",
                "呼吸系统疾病",
                "循环系统疾病",
                "恶性肿瘤情况",
                "消化系统疾病",
                "神经系统疾病",
                "泌尿生殖系统疾病",
                "眼耳鼻喉相关疾病",
                "内分泌及免疫系统疾病",
            ],
            "sywu": ["label", "entity"],
        }
        if not isinstance(answer_json, dict):
            answer_json = {"sywu": answer_json}
        if not isinstance(generated_answer, dict):
            generated_answer = {"sywu": generated_answer}
        # 单元层级
        ce = ie = me = se = 0  # 初始化计数器：正确提取、错误提取、漏提取、误提取
        precision = 0
        recall = 0
        error_keys = []
        try:
            for unit_name, answer_unit_value in answer_json.items():
                generate_unit_value = generated_answer.get(unit_name)
                # 检查 generate_unit_value 是否为空列表
                if generate_unit_value == [] and len(answer_unit_value)==1:
                    # 检查 answer_unit_value 中的所有值是否都为 "NA"
                    if all(value == "NA" for value in answer_unit_value.values()):
                        precision = 1
                        recall = 1
                        print(f"{unit_name}全部为NA，准确率为1，召回率为1")
                        continue
                if generate_unit_value is None:
                    me += len(answer_unit_value) * len(unit_loc_mapping[unit_name])
                    continue
                length_gap = len(answer_unit_value) - len(generate_unit_value)
                if length_gap > 0: # 生成的数据更少
                    me += length_gap * len(unit_loc_mapping[unit_name])
                    print(f"{unit_name}缺少{length_gap}条数据")
                elif length_gap < 0: # 生成的数据更多
                    se += abs(length_gap) * len(unit_loc_mapping[unit_name])
                if unit_name=='Date':
                    if isinstance(answer_unit_value,dict):
                        answer_unit_value = [answer_unit_value]
                    if isinstance(generate_unit_value,dict):
                        generate_unit_value = [generate_unit_value]
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
                    # 数据准备
                    # 先全部转换为小写
                    for k_g, v_g in list(generate_unit_value_dict.items()):
                        # 可能出现整型数字，也可能是整型字符串，做一次归一化，变成字符串
                        v_g = str(v_g) if isinstance(v_g, int) else v_g
                        # 空串空列表都映射为na
                        if v_g == "" or v_g == []:
                            v_g = "na"
                        if isinstance(v_g, list):
                            v_g = [i.lower() for i in v_g]
                        else:
                            if isinstance(v_g, str):
                                v_g = v_g.lower()
                        # 诊断医生转为拼音
                        if k_g in ["Diagnosing Doctor"]:  # 诊断医生转为拼音
                            pinyin_list_g = pinyin(v_g, style=Style.NORMAL)
                            v_g = "".join(word[0] for word in pinyin_list_g)
                        generate_unit_value_dict[k_g] = v_g  # 重新赋值以更新字典中的数据

                    for k_a, v_a in list(answer_unit_value_dict.items()):
                        # 可能出现整型数字，也可能是整型字符串，做一次归一化，变成字符串
                        v_a = str(v_a) if isinstance(v_a, int) else v_a
                        # 空串空列表都映射为na
                        if v_a == "" or v_a == []:
                            v_a = "na"
                        if isinstance(v_a, list):
                            v_a = [i.lower() for i in v_a]
                        else:
                            if isinstance(v_a, str):
                                v_a = v_a.lower()
                        # 诊断医生转为拼音
                        if k_a in ["Diagnosing Doctor"]:  # 诊断医生转为拼音
                            pinyin_list_a = pinyin(v_a, style=Style.NORMAL)
                            v_a = "".join(word[0] for word in pinyin_list_a)
                        answer_unit_value_dict[k_a] = v_a
                    # 数据准备结束

                    for k_a, v_a in answer_unit_value_dict.items():
                        v_a = v_a[0] if len(v_a) == 1 else v_a

                        if k_a not in generate_unit_value_dict: #生成数据缺失key
                            me += 1
                            error_keys.append({k_a: "key not exist in generate"})
                            continue
                        v_g = generate_unit_value_dict[k_a]

                        v_g = v_g[0] if len(v_g) == 1 else v_g
                        if isinstance(v_a, list) and isinstance(v_g, list):
                            if set(v_a) != set(v_g):
                                ie += 1
                                error_keys.append({k_a: {"answer": v_a, "generate": v_g}})
                            else:
                                ce += 1
                        elif v_a!="na" and v_g=="na":
                            me += 1
                            error_keys.append({k_a: {"answer": v_a, "generate": v_g}})
                        elif v_a != v_g:
                            ie += 1
                            error_keys.append({k_a: {"answer": v_a, "generate": v_g}})
                        elif  v_a=="na" and v_g=="na":
                            continue
                        else:
                            ce+= 1

                    # 计算 SE
                    for key in generate_unit_value_dict.keys():
                        if key not in answer_unit_value_dict.keys():
                            se += 1  # 误提取
                            error_keys.append({key: "key not exist in answer"})

                    # 计算 Precision 和 Recall
                    if ce==0 and ie==0 and me==0 and se==0:
                        precision = 1
                        recall = 1
                    else:
                        precision = ce / (ce + ie + se) if (ce + ie + se) > 0 else 0
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
            print(f"{datetime.now()}:{e}")
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
        {"entity": "呼吸肌", "label": "身体"},
        {"entity": "呼吸肌痉挛", "label": "临床表现"},
        {"entity": "呼吸困难", "label": "临床表现"},
        {"entity": "全身肌", "label": "身体"},
        {"entity": "全身肌张力高", "label": "临床表现"},
        {"entity": "颈部", "label": "身体"},
        {"entity": "颈部强硬", "label": "临床表现"},
    ]
    answer_json = [
        {"entity": "呼吸肌", "label": "身体"},
        {"entity": "颈部", "label": "身体"},
        {"entity": "颈部强硬", "label": "临床表现"},
        {"entity": "呼吸困难", "label": "临床表现"},
        {"entity": "全身肌1", "label": "身体"},
        {"entity": "全身肌张力高", "label": "临床表现"},
        {"entity": "呼吸肌痉挛", "label": "临床表现"},
    ]

    # 调用函数并打印结果
    result = f1.labor_recall_precise(answer_json, generated_answer)
    print(result)
