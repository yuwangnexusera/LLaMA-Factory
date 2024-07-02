import json
from copy import deepcopy
from difflib import SequenceMatcher
import re
import numpy as np
from typing import List, Tuple
from pypinyin import pinyin, Style
from datetime import datetime
import sys
sys.path.append(".")
from data_augmentation import prompt_dict

class F1score:

    def __init__(self):
        self.special_value_mapping = {"腺癌": ["肺腺癌", "浸润性腺癌"], "小细胞癌": ["小细胞肺癌"]}
        self.reverse_mapping = self.create_reverse_mapping(self.special_value_mapping)

    def create_reverse_mapping(self, mapping):
        reverse_mapping = {}
        for key, values in mapping.items():
            for value in values:
                reverse_mapping[value] = key
        return reverse_mapping

    def normalize_date(self, date_str):
        patterns = [
            r"(\d{4})-(\d{1,2})-(\d{1,2})",
            r"(\d{4})/(\d{1,2})/(\d{1,2})",
            r"(\d{4})\.(\d{1,2})\.(\d{1,2})",
            r"(\d{4})年(\d{1,2})月(\d{1,2})日",
            r"(\d{1,2})-(\d{1,2})",
            r"(\d{1,2})/(\d{1,2})",
            r"(\d{1,2})\.(\d{1,2})",
            r"(\d{1,2})月(\d{1,2})日",
        ]
        for pattern in patterns:
            match = re.match(pattern, date_str)
            if match:
                groups = match.groups()
                if len(groups) == 3:
                    year, month, day = groups
                    return f"{year}-{int(month):02d}-{int(day):02d}"
                elif len(groups) == 2:
                    month, day = groups
                    return f"{int(month):02d}-{int(day):02d}"
        return date_str

    def normalize_value(self, value): 
        '''针对英文数据的时候使用'''
        if isinstance(value, str):
            value = value.lower() if value !="NA" else value
            if value in self.reverse_mapping: #同义词转换
                return self.reverse_mapping[value]
            normalized_date = self.normalize_date(value)
            if normalized_date != value:
                return normalized_date
        return value

    def compare_json(self, answer, generate):
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
            return score / max_len if max_len > 0 else 1
        else:
            return 1 if self.normalize_value(answer) == self.normalize_value(generate) else 0

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

        indexed_scores.sort(reverse=True, key=lambda x: x[0])
        return indexed_scores

    def labor_recall_precise(self, generated_answer, answer_json, include_na_in_total=False):
        '''输入格式{"单元名":[]}'''
        unit_loc_mapping = prompt_dict._default_unit_locs
        if not isinstance(answer_json, dict):
            answer_json = {"sywu": answer_json}
        if not isinstance(generated_answer, dict):
            generated_answer = {"sywu": generated_answer}

        ce = ie = me = se = 0
        precision = 0
        recall = 0
        error_keys = []
        try:
            for unit_name, answer_unit_value in answer_json.items():
                generate_unit_value = generated_answer.get(unit_name)
                if generate_unit_value is None:
                    me += len(answer_unit_value) * len(unit_loc_mapping[unit_name])
                    continue
                if generate_unit_value == []:
                    generate_unit_value = [{loc:"NA" for loc in unit_loc_mapping[unit_name]}]
                    print(f"{unit_name}全部为NA")
                length_gap = len(answer_unit_value) - len(generate_unit_value)
                if length_gap > 0:
                    me += length_gap * len(unit_loc_mapping[unit_name])
                    print(f"{unit_name}缺少{length_gap}条数据")
                elif length_gap < 0:
                    se += abs(length_gap) * len(unit_loc_mapping[unit_name])
                if unit_name == "Date" or unit_name=="日期":
                    if isinstance(answer_unit_value, dict):
                        answer_unit_value = [answer_unit_value]
                    if isinstance(generate_unit_value, dict):
                        generate_unit_value = [generate_unit_value]
                similarity_tuple_list = self.sort_similarity_matrix(answer_unit_value, generate_unit_value) #计算相似度
                compared_generate_index = set() #已对比过的json对象
                compared_answer_index = set()
                for similarity, (answer_index, generate_index) in similarity_tuple_list:
                    if generate_index in compared_generate_index or answer_index in compared_answer_index:
                        continue
                    compared_generate_index.add(generate_index)
                    compared_answer_index.add(answer_index)
                    answer_unit_value_dict = answer_unit_value[answer_index]
                    generate_unit_value_dict = generate_unit_value[generate_index]
                    # 数据处理
                    for k_g, v_g in generate_unit_value_dict.items():
                        v_g = str(v_g) if isinstance(v_g, int) else v_g
                        if v_g == "" or v_g == [] or v_g == ["NA"]:
                            v_g = "NA"
                        if isinstance(v_g, list):
                            v_g = list(map(self.normalize_value, v_g))
                        else:
                            v_g = self.normalize_value(v_g)
                        if k_g in ["Diagnosing Doctor"]: #诊断医生的拼音转换
                            pinyin_list_g = pinyin(v_g, style=Style.NORMAL)
                            v_g = "".join(word[0] for word in pinyin_list_g)
                        generate_unit_value_dict[k_g] = v_g.strip() if isinstance(v_g, str) else v_g

                    for k_a, v_a in answer_unit_value_dict.items():
                        v_a = str(v_a) if isinstance(v_a, int) else v_a
                        if v_a == "" or v_a == [] or v_a == ["NA"]:
                            v_a = "NA"
                        if isinstance(v_a, list):
                            v_a = list(map(self.normalize_value, v_a))
                        else:
                            v_a = self.normalize_value(v_a)
                        if k_a in ["Diagnosing Doctor"]:
                            pinyin_list_a = pinyin(v_a, style=Style.NORMAL)
                            v_a = "".join(word[0] for word in pinyin_list_a)
                        answer_unit_value_dict[k_a] = v_a.strip() if isinstance(v_a, str) else v_a

                    # 对比
                    for k_a, v_a in answer_unit_value_dict.items():
                        v_a = v_a[0] if isinstance(v_a, list) and len(v_a) == 1 else v_a

                        if k_a not in generate_unit_value_dict and v_a != "NA":
                            me += 1
                            error_keys.append({k_a: "key not recall in generate"})
                            continue
                        v_g = generate_unit_value_dict[k_a]

                        v_g = v_g[0] if isinstance(v_g, list) and len(v_g) == 1 else v_g
                        if isinstance(v_a, list) and isinstance(v_g, list):
                            if set(v_a) != set(v_g):
                                ie += 1
                                error_keys.append({k_a: {"answer": v_a, "generate": v_g}})
                            else:
                                ce += 1
                        elif v_a != "NA" and v_g == "NA":
                            me += 1
                            error_keys.append({k_a: {"answer": v_a, "generate": v_g}})
                        elif v_a != v_g:
                            ie += 1
                            error_keys.append({k_a: {"answer": v_a, "generate": v_g}})
                        else:
                            ce += 1

                    for key in generate_unit_value_dict.keys():
                        if key not in answer_unit_value_dict.keys():
                            se += 1
                            error_keys.append({key: "key not exist in answer,supurious!"})
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

    # 示例用法
    answer_json = {
        "治疗用药方案": [
            {"治疗开始日期": "2021-03-16", "治疗用药名称": ["阿来替尼"], "治疗用药是否为建议": "否", "治疗结束日期": "NA"},
            {
                "治疗开始日期": "2021-08-06",
                "治疗用药名称": ["培美曲塞", "顺铂"],
                "治疗用药是否为建议": "否",
                "治疗结束日期": "NA",
            },
        ],
    }
    generate_output = json.loads(
            '[{"治疗开始日期": "2021-03-16", "治疗用药名称": ["阿来替尼"], "治疗用药是否为建议": "否", "治疗结束日期": "NA"}, {"治疗开始日期": "2021-08-27", "治疗用药名称": ["培美曲塞", "顺铂"], "治疗用药是否为建议": "否", "治疗结束日期": "NA"}, {"治疗开始日期": "2021-08-06", "治疗用药名称": ["培美曲塞", "顺铂"], "治疗用药是否为建议": "否", "治疗结束日期": "NA"}]'
        )
    generated_answer = {"治疗用药方案": generate_output}

    f1 = F1score()
    result = f1.labor_recall_precise(generated_answer, answer_json)
    print(result)
