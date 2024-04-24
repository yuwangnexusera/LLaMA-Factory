import json
from copy import deepcopy
from difflib import SequenceMatcher
import re


class F1score():
    # 用于实验过程中不区分单元的比较
    def labor_recall_precise(self, generated_answer, answer_json, include_na_in_total=False):
        """
        CE (Correct Extraction) ↔ TP (True Positive)
        IE (Incorrect Extraction) + SE (Spurious Extraction) ↔ FP (False Positive)
        ME (Missed Extraction) ↔ FN (False Negative)
        TN (True Negative): 在信息提取任务中无法定义。
        """
        unit_loc_mapping = {'Basic Information': ['Date of Birth', 'Age', 'Gender'], 'Disease': ['Date of First Diagnosis', 'Time of First Pathological Diagnosis (Biopsy, Post-operative Pathology, etc.)', 'Time of First Lung Resection', 'Time of First Imaging Diagnosis', 'Time of First Treatment (Drugs, Radiotherapy, etc.)', 'Time of First Symptom', 'Disease Name'], 'Symptom': ['ECOG Score', 'ECOG Date'], 'Diagnosis': ['Diagnosing Doctor'], 'Imaging': ['Brain Metastasis Date', 'Brain Metastasis Site'], 'Pathology': ['Pathology Date', 'Pathology Type'], 'Genetic Testing': ['ALK', 'MET', 'RB1', 'RET', 'BRAF', 'BRCA', 'EGFR', 'FGFR', 'KRAS', 'NTRK', 'ROS1', 'TP53', 'KEAP1', 'STK11', 'HER2 (ERBB2)', 'HER3 (ERBB3)', 'HER4 (ERBB4)', 'Genetic Testing Date'], 'Immune Testing': ['Immune Cell', 'Combined Positive Score', 'Tumor Proportion Score', 'PD-L1', 'Immunological Test Date'], 'Cancer treatment': ['Surgical Site', 'Treatment Start Date', 'Treatment Drug Names', 'Treatment End Date', 'Specific Tumor Treatment Method'], 'Treatment Drug Plan': ['Treatment Start Date', 'Treatment Drug Names', 'Treatment End Date', 'Is Treatment Drug Recommended'], 'Comorbid Disease': ['Date of Confirmed Disease', 'Information Source', 'Infectious Diseases', 'Respiratory System Diseases', 'Circulatory System Diseases', 'Malignant Tumor Conditions', 'Digestive System Diseases', 'Nervous System Diseases', 'Urogenital System Diseases', 'Eye, Ear, Nose, and Throat Related Diseases', 'Endocrine and Immune System Diseases'], 'Date': ['Admission Date', 'Discharge Date', 'Medical History Collection Date', 'Record Date']}

        # 单元层级
        ce = ie = me = se = 0  # 初始化计数器：正确提取、错误提取、漏提取、误提取
        error_keys = []
        try:
            for unit_name, answer_unit_value in answer_json.items():
                if generated_answer.get(unit_name) is None and unit_name in unit_loc_mapping.keys():
                    me = len(unit_loc_mapping[unit_name]) + me
                else: #如果单元存在，对比每个点位
                    # TODO 如果是列表，如何比较
                    generate_unit_value = generated_answer.get(unit_name)
                    # 先全部转换为小写
                    for k_g, v_g in generate_unit_value.items():
                        if isinstance(v_g,list):
                            generate_unit_value[k_g] = [i.lower() for i in v_g]
                        else:
                            if isinstance(v_g,str):
                                generate_unit_value[k_g] = v_g.lower() 
                    for k_a, v_a in answer_unit_value.items():
                        if isinstance(v_a,list):
                            answer_unit_value[k_a] = [i.lower() for i in v_a]
                        else:
                            if isinstance(v_a, str):
                                answer_unit_value[k_a] = v_a.lower()
                    # 从 answer_json 获取所有可能的键作为参考
                    reference_keys = set(answer_unit_value.keys())
                    generated_keys = set(generate_unit_value.keys())

                    # 计算 CE 和 IE TODO 需优化
                    for key in reference_keys:
                        if key=="Age":
                            generate_unit_value[key] = str(generate_unit_value[key])
                            answer_unit_value[key] = str(answer_unit_value[key])
                        if generate_unit_value[key]=="":
                            generate_unit_value[key] = "NA"
                        if answer_unit_value[key]=="":
                            answer_unit_value[key] = "NA"
                        if key in ["Diagnosing Doctor"]:
                            continue
                        if key in generated_keys:
                            # 数据准备，列表+长度为1，取出这个值；列表长度不是1，转为set，比较列表。
                            if isinstance(generate_unit_value[key],list) :
                                if len(generate_unit_value[key])==1:
                                    generate_unit_value[key] = generate_unit_value[key][0]
                                else:
                                    generate_unit_value[key] = set(generate_unit_value[key])
                            if isinstance(answer_unit_value[key],list):
                                if len(answer_unit_value[key])==1:
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
                                    else "NA"
                                )
                                answer_unit_value[key] = re.findall(r"\d+",answer_unit_value[key])[0] if re.findall(r"\d+", answer_unit_value[key]) else "NA"

                            if generate_unit_value[key] == answer_unit_value[key]:
                                ce += 1  # 提取正确
                            else:
                                ie += 1  # 提取错误
                                error_keys.append({key:[answer_unit_value[key],generate_unit_value[key]]})
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
        except Exception as e:
            return {
                "correct_extraction": 0,
                "incorrect_extraction": 0,
                "missed_extraction": 0,
                "spurious_extraction": 0,
                "precision": 0,
                "recall": 0,
                "Exception":e
            }


if __name__ == "__main__":
    f1 = F1score()
    # 示例数据
    generated_answer = {
        "Basic Information": {"Date of Birth": "1952-03-17", "Age": 61, "Gender": "Unknown"},
        "Disease": {
            "Date of First Diagnosis": "2021-12-17",
            "Time of First Pathological Diagnosis (Biopsy, Post-operative Pathology, etc.)": "NA",
            "Time of First Lung Resection": "NA",
            "Time of First Imaging Diagnosis": "2020-07-31",
            "Time of First Treatment (Drugs, Radiotherapy, etc.)": "NA",
            "Time of First Symptom": "NA",
            "Disease Name": ["Mesothelioma", "Small cell carcinoma"],
        },
        "Symptom": {"ECOG Score": "NA", "ECOG Date": "NA"},
        "Diagnosis": {"Diagnosing Doctor": "NA"},
        "Imaging": {"Brain Metastasis Date": "2021-10-02", "Brain Metastasis Site": "NA"},
        "Pathology": {"Pathology Date": "NA", "Pathology Type": ["Spindle cell carcinoma"]},
        "Genetic Testing": {
            "ALK": "NA",
            "MET": ["Other Rare Mutations"],
            "RB1": ["Positive"],
            "RET": ["Rearrangement"],
            "BRAF": "NA",
            "BRCA": ["BRCA1"],
            "EGFR": "NA",
            "FGFR": ["Fusion"],
            "KRAS": "NA",
            "NTRK": ["NTRK3"],
            "ROS1": ["S1986F"],
            "TP53": ["Positive"],
            "KEAP1": "NA",
            "STK11": "NA",
            "HER2 (ERBB2)": "NA",
            "HER3 (ERBB3)": ["Positive"],
            "HER4 (ERBB4)": "NA",
            "Genetic Testing Date": "NA",
        },
        "Immune Testing": {
            "Immune Cell": "NA",
            "Combined Positive Score": "NA",
            "Tumor Proportion Score": "96%",
            "PD-L1": "77%",
            "Immunological Test Date": "2021-02-07",
        },
        "Cancer treatment": {
            "Surgical Site": "NA",
            "Treatment Start Date": "2021-11-23",
            "Treatment Drug Names": "NA",
            "Treatment End Date": "2021-06-18",
            "Specific Tumor Treatment Method": ["Particle implantation"],
        },
        "Treatment Drug Plan": {
            "Treatment Start Date": "2021-11-23",
            "Treatment Drug Names": "NA",
            "Treatment End Date": "2021-06-18",
            "Is Treatment Drug Recommended": "NA",
        },
        "Comorbid Disease": {
            "Date of Confirmed Disease": "NA",
            "Information Source": "NA",
            "Infectious Diseases": "NA",
            "Respiratory System Diseases": ["Atelectasis", "Obstructive Emphysema"],
            "Circulatory System Diseases": ["Luminal Stenosis"],
            "Malignant Tumor Conditions": ["Gastric Cancer"],
            "Digestive System Diseases": ["Fatty Liver"],
            "Nervous System Diseases": "NA",
            "Urogenital System Diseases": "NA",
            "Eye, Ear, Nose, and Throat Related Diseases": ["Keratomalacia"],
            "Endocrine and Immune System Diseases": "NA",
        },
        "Date": {
            "Admission Date": "2019-09-24",
            "Discharge Date": "2022-02-10",
            "Medical History Collection Date": "NA",
            "Record Date": "NA",
        },
    }
    answer_json = {
        "Basic Information": {"Date of Birth": "1952-03-17", "Age": 61, "Gender": "Unknown"},
        "Disease": {
            "Date of First Diagnosis": "2021-12-17",
            "Time of First Pathological Diagnosis (Biopsy, Post-operative Pathology, etc.)": "NA",
            "Time of First Lung Resection": "NA",
            "Time of First Imaging Diagnosis": "2020-07-31",
            "Time of First Treatment (Drugs, Radiotherapy, etc.)": "NA",
            "Time of First Symptom": "NA",
            "Disease Name": ["Mesothelioma", "Small cell carcinoma"],
        },
        "Symptom": {"ECOG Score": "NA", "ECOG Date": "NA"},
        "Diagnosis": {"Diagnosing Doctor": "NA"},
        "Imaging": {"Brain Metastasis Date": "2021-10-02", "Brain Metastasis Site": "NA"},
        "Pathology": {"Pathology Date": "NA", "Pathology Type": ["Spindle cell carcinoma"]},
        "Genetic Testing": {
            "ALK": "NA",
            "MET": ["Other Rare Mutations"],
            "RB1": ["Positive"],
            "RET": ["Rearrangement"],
            "BRAF": "NA",
            "BRCA": ["BRCA1"],
            "EGFR": "NA",
            "FGFR": ["Fusion"],
            "KRAS": "NA",
            "NTRK": ["NTRK3"],
            "ROS1": ["S1986F"],
            "TP53": ["Positive"],
            "KEAP1": "NA",
            "STK11": "NA",
            "HER2 (ERBB2)": "NA",
            "HER3 (ERBB3)": ["Positive"],
            "HER4 (ERBB4)": "NA",
            "Genetic Testing Date": "NA",
        },
        "Immune Testing": {
            "Immune Cell": "NA",
            "Combined Positive Score": "NA",
            "Tumor Proportion Score": "96%",
            "PD-L1": "77%",
            "Immunological Test Date": "2021-02-07",
        },
        "Cancer treatment": {
            "Surgical Site": "NA",
            "Treatment Start Date": "2021-11-23",
            "Treatment Drug Names": "NA",
            "Treatment End Date": "2021-06-18",
            "Specific Tumor Treatment Method": ["Particle implantation"],
        },
        "Treatment Drug Plan": {
            "Treatment Start Date": "2021-11-23",
            "Treatment Drug Names": "NA",
            "Treatment End Date": "2021-06-18",
            "Is Treatment Drug Recommended": "NA",
        },
        "Comorbid Disease": {
            "Date of Confirmed Disease": "NA",
            "Information Source": "NA",
            "Infectious Diseases": "NA",
            "Respiratory System Diseases": ["Atelectasis", "Obstructive Emphysema"],
            "Circulatory System Diseases": ["Luminal Stenosis"],
            "Malignant Tumor Conditions": ["Gastric Cancer"],
            "Digestive System Diseases": ["Fatty Liver"],
            "Nervous System Diseases": "NA",
            "Urogenital System Diseases": "NA",
            "Eye, Ear, Nose, and Throat Related Diseases": ["Keratomalacia"],
            "Endocrine and Immune System Diseases": "NA",
        },
        "Date": {
            "Admission Date": "2019-09-24",
            "Discharge Date": "2022-02-10",
            "Medical History Collection Date": "NA",
            "Record Date": "NA",
        },
    }

    # 调用函数并打印结果
    result = f1.labor_recall_precise(generated_answer, answer_json)
    print(result)
    # pic_std = [{"url": "https://image.yoo.la/karen-test/%E5%9F%BA%E5%9B%A0%E6%B5%8B%E8%AF%951119/%E6%B5%8B%E5%9F%BA%E5%9B%A015.jpg/keep", "data": {"肿瘤治疗": [{"入院日期": "2023-08-02", "出院日期": "NA", "治疗开始日期": "2022-10-29", "治疗用药名称": ["卡铂", "培美曲塞", "贝伐珠单抗"], "治疗结束日期": "NA", "肿瘤具体治疗方式": "化疗,抗血管,免疫"}, {"入院日期": "2023-08-02", "出院日期": "NA", "治疗开始日期": "2022-11-19", "治疗用药名称": ["卡铂", "培美曲塞", "贝伐珠单抗", "信迪利单抗"], "治疗结束日期": "NA", "肿瘤具体治疗方式": "化疗,抗血管,免疫"}, {"入院日期": "2023-08-02", "出院日期": "NA", "治疗开始日期": "2022-12-10", "治疗用药名称": ["卡铂", "培美曲塞", "贝伐珠单抗", "信迪利单抗"], "治疗结束日期": "2023-00-NA", "肿瘤具体治疗方式": "化疗,抗血管,免疫"}, {"入院日期": "2023-08-02", "出院日期": "NA", "治疗开始日期": "2022-01-03", "治疗用药名称": ["卡铂", "培美曲塞", "贝伐珠单抗", "信迪利单抗"], "治疗结束日期": "2023-00-NA", "肿瘤具体治疗方式": "化疗,抗血管,免疫"}, {"入院日期": "2023-08-02", "出院日期": "NA", "治疗开始日期": "2023-01-31", "治疗用药名称": ["卡铂", "培美曲塞", "贝伐珠单抗", "信迪利单抗"], "治疗结束日期": "NA", "肿瘤具体治疗方式": "化疗,抗血管,免疫"}, {"入院日期": "2023-08-02", "出院日期": "NA", "治疗开始日期": "2023-02-27", "治疗用药名称": ["卡铂", "培美曲塞", "贝伐珠单抗", "信迪利单抗"], "治疗结束日期": "NA", "肿瘤具体治疗方式": "化疗,抗血管,免疫"}, {"入院日期": "2023-08-02", "出院日期": "NA", "治疗开始日期": "2023-03-27", "治疗用药名称": ["顺铂", "培美曲塞", "贝伐珠单抗", "信迪利单抗"], "治疗结束日期": "NA", "肿瘤具体治疗方式": "化疗,抗血管,免疫"}, {"入院日期": "2023-08-02", "出院日期": "NA", "治疗开始日期": "2023-04-25", "治疗用药名称": ["顺铂", "培美曲塞", "贝伐珠单抗", "信迪利单抗"], "治疗结束日期": "NA", "肿瘤具体治疗方式": "化疗,抗血管,免疫"}]}}, {"url": "https://yoola1-bucket.oss-cn-zhangjiakou.aliyuncs.com/ss1000/%E5%87%BA%E5%85%A5%E9%99%A2%E8%AE%B0%E5%BD%9540/%E5%87%BA%E5%85%A5%E9%99%A2%E6%8A%A5%E5%91%8A1.jpg/keep", "data": {"肿瘤治疗": [{"入院日期": "2020-06-29", "出院日期": "2020-07-22", "治疗开始日期": "NA", "治疗用药名称": ["吉非替尼"], "治疗结束日期": "NA", "肿瘤具体治疗方式": "靶向"}]}}, {"url": "https://yoola1-bucket.oss-cn-zhangjiakou.aliyuncs.com/ss1000/%E5%87%BA%E5%85%A5%E9%99%A2%E8%AE%B0%E5%BD%9540/%E5%87%BA%E5%85%A5%E9%99%A2%E6%8A%A5%E5%91%8A5.jpg/keep", "data": {"肿瘤治疗": [{"入院日期": "2023-10-20", "出院日期": "NA", "治疗开始日期": "2023-07-07", "治疗用药名称": ["依托泊苷", "卡铂"], "治疗结束日期": "NA", "肿瘤具体治疗方式": "化疗"}, {"入院日期": "2023-10-20", "出院日期": "NA", "治疗开始日期": "2023-09-NA", "治疗用药名称": "NA", "治疗结束日期": "NA", "肿瘤具体治疗方式": "放疗"}, {"入院日期": "2023-10-20", "出院日期": "NA", "治疗开始日期": "2023-10-10", "治疗用药名称": "NA", "治疗结束日期": "NA", "肿瘤具体治疗方式": "化疗"}]}}, {"url": "https://yoola1-bucket.oss-cn-zhangjiakou.aliyuncs.com/ss1000/%E5%87%BA%E5%85%A5%E9%99%A2%E8%AE%B0%E5%BD%9540/%E5%87%BA%E5%85%A5%E9%99%A2%E8%AE%B0%E5%BD%9514.jpg/keep", "data": {"肿瘤治疗": [{"入院日期": "2022-11-14", "出院日期": "NA", "治疗开始日期": "NA", "治疗用药名称": "NA", "治疗结束日期": "NA", "肿瘤具体治疗方式": "手术"}]}}, {"url": "https://yoola1-bucket.oss-cn-zhangjiakou.aliyuncs.com/ss1000/%E5%87%BA%E5%85%A5%E9%99%A2%E8%AE%B0%E5%BD%9540/%E5%87%BA%E5%85%A5%E9%99%A2%E8%AE%B0%E5%BD%9517.jpg/keep", "data": {"肿瘤治疗": [{"入院日期": "2023-07-06", "出院日期": "2023-07-08", "治疗开始日期": "2023-07-07", "治疗用药名称": "NA", "治疗结束日期": "NA", "肿瘤具体治疗方式": "手术"}]}}, {"url": "https://yoola1-bucket.oss-cn-zhangjiakou.aliyuncs.com/ss1000/%E5%87%BA%E5%85%A5%E9%99%A2%E8%AE%B0%E5%BD%9540/%E5%87%BA%E5%85%A5%E9%99%A2%E8%AE%B0%E5%BD%9533.jpg/keep", "data": {"肿瘤治疗": [{"入院日期": "2020-06-28", "出院日期": "2020-07-14", "治疗开始日期": "2020-06-29", "治疗用药名称": "NA", "治疗结束日期": "NA", "肿瘤具体治疗方式": "放疗"}, {"入院日期": "2020-06-28", "出院日期": "2020-07-14", "治疗开始日期": "2020-07-06", "治疗用药名称": "培美曲塞,卡铂", "治疗结束日期": "NA", "肿瘤具体治疗方式": "化疗"}]}}, {"url": "https://yoola1-bucket.oss-cn-zhangjiakou.aliyuncs.com/ss1000/%E5%87%BA%E5%85%A5%E9%99%A2%E8%AE%B0%E5%BD%9540/%E5%87%BA%E5%85%A5%E9%99%A2%E8%AE%B0%E5%BD%9536.jpg/keep", "data": {"肿瘤治疗": [{"入院日期": "2021-11-11", "出院日期": "2021-11-17", "治疗开始日期": "NA", "治疗用药名称": ["EP方案", "维生素B6", "西咪替丁", "二羟丙茶碱"], "治疗结束日期": "NA", "肿瘤具体治疗方式": "化疗"}]}}, {"url": "https://yoola1-bucket.oss-cn-zhangjiakou.aliyuncs.com/ss1000/%E5%87%BA%E5%85%A5%E9%99%A2%E8%AE%B0%E5%BD%9540/%E5%87%BA%E5%85%A5%E9%99%A2%E8%AE%B0%E5%BD%9540.jpg/keep", "data": {"肿瘤治疗": [{"入院日期": "2021-06-07", "出院日期": "2021-06-18", "治疗开始日期": "2021-06-16", "治疗用药名称": "NA", "治疗结束日期": "NA", "肿瘤具体治疗方式": "手术"}]}}, {"url": "https://yoola1-bucket.oss-cn-zhangjiakou.aliyuncs.com/ss1000/%E5%87%BA%E5%85%A5%E9%99%A2%E8%AE%B0%E5%BD%9540/%E5%87%BA%E5%85%A5%E9%99%A2%E8%AE%B0%E5%BD%9527.jpg/keep", "data": {"肿瘤治疗": [{"入院日期": "NA", "出院日期": "NA", "治疗开始日期": "NA", "治疗用药名称": ["培美曲塞二钠", "顺铂"], "治疗结束日期": "NA", "肿瘤具体治疗方式": "化疗"}]}}, {"url": "https://yoola1-bucket.oss-cn-zhangjiakou.aliyuncs.com/%E9%AA%8C%E6%94%B6%E6%B5%8B%E8%AF%95/%E6%82%A3%E8%80%8511%E5%88%98%2A%E4%B8%BD/%E6%82%A3%E8%80%8511%E5%88%98%2A%E4%B8%BD/8.jpg/keep", "data": {"肿瘤治疗": [{"入院日期": "NA", "出院日期": "NA", "治疗开始日期": "2022-11-23", "治疗用药名称": ["顺铂", "恩度", "奥希替尼"], "治疗结束日期": "NA", "肿瘤具体治疗方式": "胸腔灌注,化疗,靶向"}]}}, {"url": "https://yoola1-bucket.oss-cn-zhangjiakou.aliyuncs.com/ss1000/%E5%87%BA%E5%85%A5%E9%99%A2%E8%AE%B0%E5%BD%9540/%E5%87%BA%E5%85%A5%E9%99%A2%E6%8A%A5%E5%91%8A2.jpg/keep", "data": {"肿瘤治疗": [{"入院日期": "2020-06-29", "出院日期": "NA", "手术部位": "NA", "治疗开始日期": "NA", "治疗用药名称": "NA", "治疗结束日期": "NA", "肿瘤具体治疗方式": "NA"}]}}, {"url": "https://yoola1-bucket.oss-cn-zhangjiakou.aliyuncs.com/ss1000/%E5%87%BA%E5%85%A5%E9%99%A2%E8%AE%B0%E5%BD%9540/%E5%87%BA%E5%85%A5%E9%99%A2%E6%8A%A5%E5%91%8A3.jpg/keep", "data": {"肿瘤治疗": [{"入院日期": "2020-09-08", "出院日期": "2020-09-11", "手术部位": "NA", "治疗开始日期": "NA", "治疗用药名称": ["奥希替尼", "贝伐珠单抗"], "治疗结束日期": "NA", "肿瘤具体治疗方式": "靶向,抗血管"}]}}, {"url": "https://yoola1-bucket.oss-cn-zhangjiakou.aliyuncs.com/ss1000/%E5%87%BA%E5%85%A5%E9%99%A2%E8%AE%B0%E5%BD%9540/%E5%87%BA%E5%85%A5%E9%99%A2%E6%8A%A5%E5%91%8A6.jpg/keep", "data": {"肿瘤治疗": [{"入院日期": "NA", "出院日期": "NA", "手术部位": "NA", "治疗开始日期": "NA", "治疗用药名称": "NA", "治疗结束日期": "NA", "肿瘤具体治疗方式": "NA"}]}}, {"url": "https://yoola1-bucket.oss-cn-zhangjiakou.aliyuncs.com/ss1000/%E5%87%BA%E5%85%A5%E9%99%A2%E8%AE%B0%E5%BD%9540/%E5%87%BA%E5%85%A5%E9%99%A2%E8%AE%B0%E5%BD%9510.jpg/keep", "data": {"肿瘤治疗": [{"入院日期": "2023-10-18", "出院日期": "NA", "手术部位": "NA", "治疗开始日期": "NA", "治疗用药名称": "NA", "治疗结束日期": "NA", "肿瘤具体治疗方式": "NA"}]}}, {"url": "https://yoola1-bucket.oss-cn-zhangjiakou.aliyuncs.com/ss1000/%E5%87%BA%E5%85%A5%E9%99%A2%E8%AE%B0%E5%BD%9540/%E5%87%BA%E5%85%A5%E9%99%A2%E8%AE%B0%E5%BD%9514.jpg/keep", "data": {"肿瘤治疗": [{"入院日期": "2022-11-14", "出院日期": "NA", "手术部位": "其他", "治疗开始日期": "NA", "治疗用药名称": "NA", "治疗结束日期": "NA", "肿瘤具体治疗方式": "手术"}]}}, {"url": "https://yoola1-bucket.oss-cn-zhangjiakou.aliyuncs.com/ss1000/%E5%87%BA%E5%85%A5%E9%99%A2%E8%AE%B0%E5%BD%9540/%E5%87%BA%E5%85%A5%E9%99%A2%E8%AE%B0%E5%BD%9516.jpg/keep", "data": {"肿瘤治疗": [{"入院日期": "NA", "出院日期": "NA", "手术部位": "NA", "治疗开始日期": "NA", "治疗用药名称": "NA", "治疗结束日期": "NA", "肿瘤具体治疗方式": "NA"}]}}, {"url": "https://yoola1-bucket.oss-cn-zhangjiakou.aliyuncs.com/ss1000/%E5%87%BA%E5%85%A5%E9%99%A2%E8%AE%B0%E5%BD%9540/%E5%87%BA%E5%85%A5%E9%99%A2%E8%AE%B0%E5%BD%9517.jpg/keep", "data": {"肿瘤治疗": [{"入院日期": "2023-07-06", "出院日期": "2023-07-08", "手术部位": "其他", "治疗开始日期": "NA", "治疗用药名称": "NA", "治疗结束日期": "NA", "肿瘤具体治疗方式": "手术"}]}}, {"url": "https://yoola1-bucket.oss-cn-zhangjiakou.aliyuncs.com/ss1000/%E5%87%BA%E5%85%A5%E9%99%A2%E8%AE%B0%E5%BD%9540/%E5%87%BA%E5%85%A5%E9%99%A2%E8%AE%B0%E5%BD%9518.jpg/keep", "data": {"肿瘤治疗": [{"入院日期": "2022-01-04", "出院日期": "NA", "手术部位": "NA", "治疗开始日期": "NA", "治疗用药名称": "NA", "治疗结束日期": "NA", "肿瘤具体治疗方式": "NA"}]}}, {"url": "https://yoola1-bucket.oss-cn-zhangjiakou.aliyuncs.com/ss1000/%E5%87%BA%E5%85%A5%E9%99%A2%E8%AE%B0%E5%BD%9540/%E5%87%BA%E5%85%A5%E9%99%A2%E8%AE%B0%E5%BD%9519.jpg/keep", "data": {"肿瘤治疗": [{"入院日期": "2022-11-14", "出院日期": "NA", "手术部位": "其他", "治疗开始日期": "NA", "治疗用药名称": "NA", "治疗结束日期": "NA", "肿瘤具体治疗方式": "手术"}]}}, {"url": "https://yoola1-bucket.oss-cn-zhangjiakou.aliyuncs.com/ss1000/%E5%87%BA%E5%85%A5%E9%99%A2%E8%AE%B0%E5%BD%9540/%E5%87%BA%E5%85%A5%E9%99%A2%E8%AE%B0%E5%BD%9520.jpg/keep", "data": {"肿瘤治疗": [{"入院日期": "2021-12-12", "出院日期": "2021-12-23", "手术部位": "NA", "治疗开始日期": "NA", "治疗用药名称": "NA", "治疗结束日期": "NA", "肿瘤具体治疗方式": "NA"}]}}]

    # pic_gpt = [{"url": "https://image.yoo.la/karen-test/%E5%9F%BA%E5%9B%A0%E6%B5%8B%E8%AF%951119/%E6%B5%8B%E5%9F%BA%E5%9B%A015.jpg/keep", "data": {"肿瘤治疗": [{"入院日期": "2023-08-02", "出院日期": "NA", "手术部位": "NA", "治疗开始日期": "2022-10-29", "治疗用药名称": ["卡铂", "培美曲塞", "贝伐珠单抗", "信迪利单抗"], "治疗结束日期": "NA", "肿瘤具体治疗方式": "化疗,抗血管,免疫"}, {"入院日期": "2023-08-02", "出院日期": "NA", "手术部位": "NA", "治疗开始日期": "2022-11-19", "治疗用药名称": ["卡铂", "培美曲塞", "贝伐珠单抗", "信迪利单抗"], "治疗结束日期": "NA", "肿瘤具体治疗方式": "化疗,抗血管,免疫"}, {"入院日期": "2023-08-02", "出院日期": "NA", "手术部位": "NA", "治疗开始日期": "2022-12-10", "治疗用药名称": ["卡铂", "培美曲塞", "贝伐珠单抗", "信迪利单抗"], "治疗结束日期": "NA", "肿瘤具体治疗方式": "化疗,抗血管,免疫"}, {"入院日期": "2023-08-02", "出院日期": "NA", "手术部位": "NA", "治疗开始日期": "2023-01-31", "治疗用药名称": ["卡铂", "培美曲塞", "贝伐珠单抗", "信迪利单抗"], "治疗结束日期": "NA", "肿瘤具体治疗方式": "化疗,抗血管,免疫"}, {"入院日期": "2023-08-02", "出院日期": "NA", "手术部位": "NA", "治疗开始日期": "2023-02-27", "治疗用药名称": ["卡铂", "培美曲塞", "贝伐珠单抗", "信迪利单抗"], "治疗结束日期": "NA", "肿瘤具体治疗方式": "化疗,抗血管,免疫"}, {"入院日期": "2023-08-02", "出院日期": "NA", "手术部位": "NA", "治疗开始日期": "2023-03-27", "治疗用药名称": "NA", "治疗结束日期": "NA", "肿瘤具体治疗方式": "NA"}, {"入院日期": "2023-08-02", "出院日期": "NA", "手术部位": "NA", "治疗开始日期": "2023-04-25", "治疗用药名称": ["顺铂", "培美曲塞", "贝伐珠单抗", "信迪利单抗"], "治疗结束日期": "NA", "肿瘤具体治疗方式": "化疗,抗血管,免疫"}]}}, {"url": "https://yoola1-bucket.oss-cn-zhangjiakou.aliyuncs.com/ss1000/%E5%87%BA%E5%85%A5%E9%99%A2%E8%AE%B0%E5%BD%9540/%E5%87%BA%E5%85%A5%E9%99%A2%E6%8A%A5%E5%91%8A1.jpg/keep", "data": {"肿瘤治疗": {"入院日期": "2020-06-29", "出院日期": "2020-07-22", "手术部位": "NA", "治疗开始日期": "NA", "治疗用药名称": ["吉非替尼"], "治疗结束日期": "NA", "肿瘤具体治疗方式": "靶向"}}}, {"url": "https://yoola1-bucket.oss-cn-zhangjiakou.aliyuncs.com/ss1000/%E5%87%BA%E5%85%A5%E9%99%A2%E8%AE%B0%E5%BD%9540/%E5%87%BA%E5%85%A5%E9%99%A2%E6%8A%A5%E5%91%8A5.jpg/keep", "data": {"肿瘤治疗": [{"入院日期": "2023-10-20", "出院日期": "NA", "手术部位": "NA", "治疗开始日期": "2023-07-07", "治疗用药名称": ["依托泊苷", "卡铂"], "治疗结束日期": "NA", "肿瘤具体治疗方式": "化疗"}, {"入院日期": "2023-10-20", "出院日期": "NA", "手术部位": "NA", "治疗开始日期": "2023-09-NA", "治疗用药名称": "NA", "治疗结束日期": "NA", "肿瘤具体治疗方式": "放疗"}, {"入院日期": "2023-10-20", "出院日期": "NA", "手术部位": "NA", "治疗开始日期": "2023-10-10", "治疗用药名称": "NA", "治疗结束日期": "NA", "肿瘤具体治疗方式": "同步放化疗"}]}}, {"url": "https://yoola1-bucket.oss-cn-zhangjiakou.aliyuncs.com/ss1000/%E5%87%BA%E5%85%A5%E9%99%A2%E8%AE%B0%E5%BD%9540/%E5%87%BA%E5%85%A5%E9%99%A2%E8%AE%B0%E5%BD%9514.jpg/keep", "data": {"肿瘤治疗": {}}}, {"url": "https://yoola1-bucket.oss-cn-zhangjiakou.aliyuncs.com/ss1000/%E5%87%BA%E5%85%A5%E9%99%A2%E8%AE%B0%E5%BD%9540/%E5%87%BA%E5%85%A5%E9%99%A2%E8%AE%B0%E5%BD%9517.jpg/keep", "data": {"肿瘤治疗": {"入院日期": "2023-07-06", "出院日期": "2023-07-08", "手术部位": "其他", "治疗开始日期": "NA", "治疗用药名称": "NA", "治疗结束日期": "NA", "肿瘤具体治疗方式": "手术"}}}, {"url": "https://yoola1-bucket.oss-cn-zhangjiakou.aliyuncs.com/ss1000/%E5%87%BA%E5%85%A5%E9%99%A2%E8%AE%B0%E5%BD%9540/%E5%87%BA%E5%85%A5%E9%99%A2%E8%AE%B0%E5%BD%9533.jpg/keep", "data": {"肿瘤治疗": [{"入院日期": "2020-06-28", "出院日期": "2020-07-14", "手术部位": "NA", "治疗开始日期": "2020-06-29", "治疗用药名称": ["培美曲塞", "卡铂"], "治疗结束日期": "NA", "肿瘤具体治疗方式": "放疗,化疗"}]}}, {"url": "https://yoola1-bucket.oss-cn-zhangjiakou.aliyuncs.com/ss1000/%E5%87%BA%E5%85%A5%E9%99%A2%E8%AE%B0%E5%BD%9540/%E5%87%BA%E5%85%A5%E9%99%A2%E8%AE%B0%E5%BD%9536.jpg/keep", "data": {"肿瘤治疗": {"入院日期": "2021-11-11", "出院日期": "2021-11-17", "手术部位": "NA", "治疗开始日期": "NA", "治疗用药名称": ["EP方案", "维生素B6", "西咪替丁", "二羟丙茶碱"], "治疗结束日期": "NA", "肿瘤具体治疗方式": "化疗"}}}, {"url": "https://yoola1-bucket.oss-cn-zhangjiakou.aliyuncs.com/ss1000/%E5%87%BA%E5%85%A5%E9%99%A2%E8%AE%B0%E5%BD%9540/%E5%87%BA%E5%85%A5%E9%99%A2%E8%AE%B0%E5%BD%9540.jpg/keep", "data": {"肿瘤治疗": {"入院日期": "2021-06-07", "出院日期": "2021-06-18", "手术部位": "切肺", "治疗开始日期": "2021-06-16", "治疗用药名称": "NA", "治疗结束日期": "NA", "肿瘤具体治疗方式": "手术"}}}, {"url": "https://yoola1-bucket.oss-cn-zhangjiakou.aliyuncs.com/ss1000/%E5%87%BA%E5%85%A5%E9%99%A2%E8%AE%B0%E5%BD%9540/%E5%87%BA%E5%85%A5%E9%99%A2%E8%AE%B0%E5%BD%9527.jpg/keep", "data": {"肿瘤治疗": {"入院日期": "NA", "出院日期": "NA", "手术部位": "NA", "治疗开始日期": "NA", "治疗用药名称": ["培美曲塞", "顺铂"], "治疗结束日期": "NA", "肿瘤具体治疗方式": "化疗"}}}, {"url": "https://yoola1-bucket.oss-cn-zhangjiakou.aliyuncs.com/%E9%AA%8C%E6%94%B6%E6%B5%8B%E8%AF%95/%E6%82%A3%E8%80%8511%E5%88%98%2A%E4%B8%BD/%E6%82%A3%E8%80%8511%E5%88%98%2A%E4%B8%BD/8.jpg/keep", "data": {"肿瘤治疗": [{"入院日期": "NA", "出院日期": "NA", "手术部位": "NA", "治疗开始日期": "2022-11-23", "治疗用药名称": ["顺铂", "恩度", "奥希替尼"], "治疗结束日期": "NA", "肿瘤具体治疗方式": "胸腔灌注,化疗,靶向"}]}}, {"url": "https://yoola1-bucket.oss-cn-zhangjiakou.aliyuncs.com/ss1000/%E5%87%BA%E5%85%A5%E9%99%A2%E8%AE%B0%E5%BD%9540/%E5%87%BA%E5%85%A5%E9%99%A2%E6%8A%A5%E5%91%8A2.jpg/keep", "data": {"肿瘤治疗": {"入院日期": "2020-06-29", "出院日期": "NA", "手术部位": "NA", "治疗开始日期": "NA", "治疗用药名称": "NA", "治疗结束日期": "NA", "肿瘤具体治疗方式": "NA"}}}, {"url": "https://yoola1-bucket.oss-cn-zhangjiakou.aliyuncs.com/ss1000/%E5%87%BA%E5%85%A5%E9%99%A2%E8%AE%B0%E5%BD%9540/%E5%87%BA%E5%85%A5%E9%99%A2%E6%8A%A5%E5%91%8A3.jpg/keep", "data": {"肿瘤治疗": {"入院日期": "2020-09-08", "出院日期": "2020-09-11", "手术部位": "NA", "治疗开始日期": "NA", "治疗用药名称": ["奥希替尼", "安罗替尼", "贝伐珠单抗"], "治疗结束日期": "NA", "肿瘤具体治疗方式": "靶向,抗血管"}}}, {"url": "https://yoola1-bucket.oss-cn-zhangjiakou.aliyuncs.com/ss1000/%E5%87%BA%E5%85%A5%E9%99%A2%E8%AE%B0%E5%BD%9540/%E5%87%BA%E5%85%A5%E9%99%A2%E6%8A%A5%E5%91%8A6.jpg/keep", "data": {"肿瘤治疗": [{"入院日期": "NA", "出院日期": "NA", "手术部位": "NA", "治疗开始日期": "NA", "治疗用药名称": "NA", "治疗结束日期": "NA", "肿瘤具体治疗方式": "NA"}]}}, {"url": "https://yoola1-bucket.oss-cn-zhangjiakou.aliyuncs.com/ss1000/%E5%87%BA%E5%85%A5%E9%99%A2%E8%AE%B0%E5%BD%9540/%E5%87%BA%E5%85%A5%E9%99%A2%E8%AE%B0%E5%BD%9510.jpg/keep", "data": {"肿瘤治疗": {"入院日期": "2023-10-18", "出院日期": "NA", "手术部位": "NA", "治疗开始日期": "NA", "治疗用药名称": "NA", "治疗结束日期": "NA", "肿瘤具体治疗方式": "NA"}}}, {"url": "https://yoola1-bucket.oss-cn-zhangjiakou.aliyuncs.com/ss1000/%E5%87%BA%E5%85%A5%E9%99%A2%E8%AE%B0%E5%BD%9540/%E5%87%BA%E5%85%A5%E9%99%A2%E8%AE%B0%E5%BD%9514.jpg/keep", "data": {"肿瘤治疗": {"入院日期": "2022-11-14", "出院日期": "NA", "手术部位": "切肺", "治疗开始日期": "NA", "治疗用药名称": "NA", "治疗结束日期": "NA", "肿瘤具体治疗方式": "手术"}}}, {"url": "https://yoola1-bucket.oss-cn-zhangjiakou.aliyuncs.com/ss1000/%E5%87%BA%E5%85%A5%E9%99%A2%E8%AE%B0%E5%BD%9540/%E5%87%BA%E5%85%A5%E9%99%A2%E8%AE%B0%E5%BD%9516.jpg/keep", "data": {"肿瘤治疗": {"入院日期": "NA", "出院日期": "NA", "手术部位": "NA", "治疗开始日期": "NA", "治疗用药名称": "NA", "治疗结束日期": "NA", "肿瘤具体治疗方式": "NA"}}}, {"url": "https://yoola1-bucket.oss-cn-zhangjiakou.aliyuncs.com/ss1000/%E5%87%BA%E5%85%A5%E9%99%A2%E8%AE%B0%E5%BD%9540/%E5%87%BA%E5%85%A5%E9%99%A2%E8%AE%B0%E5%BD%9517.jpg/keep", "data": {"肿瘤治疗": {"入院日期": "2023-07-06", "出院日期": "2023-07-08", "手术部位": "其他", "治疗开始日期": "NA", "治疗用药名称": "NA", "治疗结束日期": "NA", "肿瘤具体治疗方式": "手术"}}}, {"url": "https://yoola1-bucket.oss-cn-zhangjiakou.aliyuncs.com/ss1000/%E5%87%BA%E5%85%A5%E9%99%A2%E8%AE%B0%E5%BD%9540/%E5%87%BA%E5%85%A5%E9%99%A2%E8%AE%B0%E5%BD%9518.jpg/keep", "data": {"肿瘤治疗": {"入院日期": "2022-01-04", "出院日期": "NA", "手术部位": "NA", "治疗开始日期": "NA", "治疗用药名称": "NA", "治疗结束日期": "NA", "肿瘤具体治疗方式": "NA"}}}, {"url": "https://yoola1-bucket.oss-cn-zhangjiakou.aliyuncs.com/ss1000/%E5%87%BA%E5%85%A5%E9%99%A2%E8%AE%B0%E5%BD%9540/%E5%87%BA%E5%85%A5%E9%99%A2%E8%AE%B0%E5%BD%9519.jpg/keep", "data": {"肿瘤治疗": {"入院日期": "2022-11-14", "出院日期": "NA", "手术部位": "NA", "治疗开始日期": "NA", "治疗用药名称": "NA", "治疗结束日期": "NA", "肿瘤具体治疗方式": "手术"}}}, {"url": "https://yoola1-bucket.oss-cn-zhangjiakou.aliyuncs.com/ss1000/%E5%87%BA%E5%85%A5%E9%99%A2%E8%AE%B0%E5%BD%9540/%E5%87%BA%E5%85%A5%E9%99%A2%E8%AE%B0%E5%BD%9520.jpg/keep", "data": {"肿瘤治疗": {"入院日期": "2021-12-12", "出院日期": "2021-12-23", "手术部位": "NA", "治疗开始日期": "NA", "治疗用药名称": "NA", "治疗结束日期": "NA", "肿瘤具体治疗方式": "NA"}}}]
    # precise, recall, diffs = compare_json_data(pic_std, pic_gpt, "肿瘤治疗")
    # # from MergeOutputUtil import DictMerger
    # # res = DictMerger([]).process_repetitive_data(standard_json["疾病"])
    # print(precise, recall, diffs)
