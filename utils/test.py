import json
import sys
sys.path.append(".")
class F1score:
    # 用于实验过程中不区分单元的比较
    def labor_recall_precise(self, generated_answer, answer_json, include_na_in_total=False):
        """
        CE (Correct Extraction) ↔ TP (True Positive)
        IE (Incorrect Extraction) + SE (Spurious Extraction) ↔ FP (False Positive)
        ME (Missed Extraction) ↔ FN (False Negative)
        TN (True Negative): 在信息提取任务中无法定义。
        """
        ce = ie = me = se = 0  # 初始化计数器：正确提取、错误提取、漏提取、误提取

        # 从 answer_json 获取所有可能的键作为参考
        reference_keys = set(answer_json.keys())
        generated_keys = set(generated_answer.keys())

        # 计算 CE 和 IE
        for key in reference_keys:
            if key in generated_keys:
                # 相等，或者一个列表中只有一个元素均可以算正确 generated_answer[key]为字符串 generated_answer[key]为包含一个元素的列表
                if isinstance(generated_answer[key], list) and len(generated_answer[key]) == 1:
                    generated_answer[key] = generated_answer[key][0]
                if isinstance(answer_json[key], list) and len(answer_json[key]) == 1:
                    answer_json[key] = answer_json[key][0]
                if generated_answer[key] == answer_json[key]:
                    ce += 1  # 提取正确
                else:
                    ie += 1  # 提取错误
            else:
                me += 1  # 漏提取

        # 计算 SE
        for key in generated_keys:
            if key not in reference_keys:
                se += 1  # 误提取

        # 计算 Precision 和 Recall
        precision = ce / (ce + ie) if (ce + ie) > 0 else 0
        recall = ce / (ce + ie + me) if (ce + ie + me) > 0 else 0

        return {
            "correct_extraction": ce,
            "incorrect_extraction": ie,
            "missed_extraction": me,
            "spurious_extraction": se,
            "precision": precision,
            "recall": recall,
        }


if __name__ == "__main__":
    res = {
        "ALK": ["Amplification"],
        "ROS1": ["S1986F"],
        "RET": ["Insert"],
        "Date of first diagnosis": "2019-02-16",
        "Date of treatment start": "2019-12-21",
        "PDL1": "50%",
        "Date of brain metastasis": "2020-12-06",
        "Date of first imaging diagnosis": "2018-01-21",
        "Date of birth": "1970-07-07",
        "Date of genetic testing": "2019-03-26",
        "Date of treatment end": "2024-06-16",
        "ECOG date": "2021-08-23",
        "Date of discharge": "2024-05-25",
        "Name of treatment drugs": [],
        "Pathological type": ["Adenocarcinoma in situ"],
        "HER2(ERBB2)": ["Non-20 insertion"],
        "BRCA": ["BRCA2"],
        "Circulatory system diseases": ["Pulmonary edema"],
        "Disease name": ["Non-small cell lung cancer"],
        "STK11": ["Positive"],
        "KEAP1": ["Positive"],
        "Record date": "2021-09-29",
        "Eye, ear, nose and throat related diseases": ["Dry eye"],
        "KRAS": ["G12V"],
        "MET": ["MET14 skip"],
        "FGFR": ["FGFR2"],
        "IC": "2",
        "First lung resection time": "2018-12-16",
        "Specific tumor treatment methods": ["Targeted"],
        "Gender": "Unknown",
        "BRAF": ["non-V600"],
        "TPS": "12%",
        "Immunoassay Date": "2022-08-23",
        "Nervous System Diseases": ["Inflammation"],
        "EGFR": ["19 Del", "18 Del"],
    }
    res2 = json.loads(
        '{"ALK": ["Amplification", "Positive"], "Date of brain metastasis": "2020-12-06", "ROS1": "S1986F", "Time of first lung resection": "2018-12-16", "RET": ["Insertion", "Point mutation"], "Pathology date": "2022-08-23", "STK11": "Positive", "Name of treatment medication": "ALK inhibitor", "Time of first treatment (drugs, radiotherapy, etc.)": "2018-01-21", "Time of first symptoms": "2023-08-11", "TPS": "50%", "PDL1": "50%", "Date of gene testing": "2019-03-26", "EGFR": ["19del", "18del"], "BRAF": ["non-V600", "Positive"], "BRCA": ["BRCA2", "Positive"], "HER2(ERBB2)": "Non-20 insertion", "Gender": "Unknown", "FGFR": ["Fusion", "FGFR2"], "Date of Birth": "1974-01-17", "Eye, Ear, Nose and Throat Related Diseases": "Dry Eye", "Specific Tumor Treatment": "Targeted", "Record Date": "2024-05-25", "Diagnostic Doctor": "Zhang Hua", "Treatment Start Date": "2018-12-21", "Surgical Site": "Liver Metastasis", "Respiratory Diseases": "Pulmonary Edema", "First Imaging Diagnosis Time": "2019-02-16", "IC": "20%", "MET": ["MET14 Jump Mutation", "Other Rare Mutations"], "Discharge Date": "2024-06-16", "KEAP1": "Positive", "Date of Medical History Collection": "2018-09-19", "Date of First Diagnosis of Disease": "2024-11-17", "KRAS": "G12V", "Time of first pathological diagnosis (puncture, postoperative pathology, etc.)": "2023-02-27", "Age": "50 years old", "Urogenital system disease": "Urogenital system inflammation"}'
    )
    f1 = F1score()
    print(f1.labor_recall_precise(res, res2))
