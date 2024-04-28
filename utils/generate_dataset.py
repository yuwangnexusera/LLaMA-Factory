from db_unit_dataset import *
from playhouse.shortcuts import (
    model_to_dict,
)
import random
import google_translate
import sys

sys.path.append(".")


def split_dataset(
    dataset_list,
    ratios,
):

    if sum(ratios) != 1:
        raise ValueError("The sum of the ratios must be equal to 1.")

    # Shuffle the dataset to ensure randomness
    random.shuffle(dataset_list)

    # Calculate split indices
    first_split_idx = int(len(dataset_list) * ratios[0])
    second_split_idx = first_split_idx + int(len(dataset_list) * ratios[1])

    # Split the dataset
    first_part = dataset_list[:first_split_idx]
    second_part = dataset_list[first_split_idx:second_split_idx]
    third_part = dataset_list[second_split_idx:]

    return (
        first_part,
        second_part,
        third_part,
    )


# 打开文件准备写入
def generate_other(
    jsonl_path,
    records_80_percent,
):
    with open(
        jsonl_path,
        "a",
        encoding="utf-8",
    ) as file:
        for item in records_80_percent:

            category = ss_report_type.select(ss_report_type.report_type).where(ss_report_type.id == item.report_id)[0].report_type
            content = item.content  # 假设每一项都有.content属性
            if not content:
                content_obj = hx_ocr_result.select().where(hx_ocr_result.url == item.url)[0]
                content = content_obj.ocr_result["data"]["ocr_align"]
            # 假设system_msg是一个你需要定义的变量
            train_validation_data = {
                "conversations": [
                    {
                        "from": "human",
                        "value": f"Your task is to determine the type of medical report entered. The report type options are: ['Other', 'Examination Record', 'Doctor's Order', 'Prescription', 'Injection', 'Expense', 'Physical Examination Report' ', 'Gene testing', 'Surgery record', 'Examination record', 'Treatment record', 'Pathology report', 'Disease course record', 'Outpatient medical record', 'Discharge and admission record', 'Informed consent', ' Other consultation records', 'Pathological consultation records', 'Other disease diagnosis certificates', 'Outpatient disease diagnosis certificates', 'Inpatient and admission disease diagnosis certificates']\nMedical report: {content}\nOutput format: directly select the most appropriate item from the options and output it, without explaining or outputting redundant content",
                    },
                    {
                        "from": "assistant",
                        "value": f"{category}",
                    },
                ]
            }
            file.write(
                json.dumps(
                    train_validation_data,
                    ensure_ascii=False,
                )
                + "\n"
            )
            file.flush()

    print(jsonl_path)
def insert_loc_in_answer(data,unit_name,loc,val):
    for d in data:
        output = json.loads(d["output"])
        for unit,vals in output.items():
            if unit == unit_name:
                for v in vals:
                    v[loc] = val
        d["output"] = json.dumps(output, ensure_ascii=False)
    return data

# 从增强数据到训练数据 test
def generate_extract_dataset(
    aug_data_path,
    train_data_path,
):
    with open(aug_data_path, "r", encoding="utf-8") as file, open(train_data_path, "a", encoding="utf-8") as train_file:
        datasets_json = json.load(file)
        converted_data = []
        for dataset in datasets_json:
            input_val = dataset.get("input", "")
            if not input_val:
                continue
            dataset.pop("input", None)
            converted_data.append(
                {
                    "instruction": "你的任务是从输入报告中提取医学信息，并以json格式输出，输入报告：",
                    "input": input_val + "json结果：",
                    "output": json.dumps(dataset, ensure_ascii=False),
                }
            )
        train_file.write(json.dumps(converted_data, ensure_ascii=False) + "\n")

        train_file.flush()


def flatten_nested_list(nested_list):
    flat_list = []
    for item in nested_list:
        if isinstance(item, list):  # 如果元素是列表，递归调用
            flat_list.extend(flatten_nested_list(item))
        else:
            flat_list.append(item)  # 如果元素不是列表，直接添加到结果列表
    return flat_list


def translate_zh_dataset(file_name):
    en_dataset = []
    with open("utils/mapping_answer_zh_en.json", "r", encoding="utf-8") as f:
        mapping_zh_en = json.load(f)
    with open(file_name, "r", encoding="utf-8") as zh_file, open(
        file_name.replace("_zh", "_en"), "w", encoding="utf-8"
    ) as en_file:
        data = json.load(zh_file)
        i = 1
        for ds_item in data:
            output = {}
            i = i + 1
            print(len(ds_item["input"]), "----", i)
            instruction = (
                "Your task is to extract medical information from the input report and output it in JSON format. Input report:"
            )
            input = google_translate.translate_text(ds_item["input"])
            if not input:
                print("error input")
            output_zh = json.loads(ds_item["output"])["output"]
            for key, value in output_zh.items():
                if value=="NA":
                    continue
                new_key = mapping_zh_en.get(key, key)
                if isinstance(value, list):
                    value = flatten_nested_list(value)
                    output[new_key] = []
                    for v in value:
                        if key in ["治疗用药名称"]:
                            new_value = google_translate.translate_text(v)
                        else:
                            new_value = mapping_zh_en.get(v, v)
                        output[new_key].append(new_value)
                else:
                    if key in ["治疗用药名称"]:
                        new_value = google_translate.translate_text(value)
                    else:
                        new_value = mapping_zh_en.get(value, value)
                    output[new_key] = new_value
            en_dataset.append({"instruction": instruction, "input": input, "output": json.dumps(output, ensure_ascii=False)})

        en_file.write(json.dumps(en_dataset, indent=4, ensure_ascii=False))

        en_file.flush()


def fill_NA_answer(file_name):
    dict_list = set()
    with open(file_name, "r", encoding="utf-8") as file:
        json_data = json.load(file)

    for item in json_data:
        try:
            asnwer = json.loads(item["output"])
            dict_list.update(asnwer.keys())
            # 长度不一致,可能翻译时key的名字没对上，不使用这个mappings，用从数据集中取出的所有key
        except Exception as e:
            print(e)
            print(item["output"])
    dict_list = list(dict_list)
    new_list = []
    for item in json_data:
        output_data = {}
        try:
            output_data = json.loads(item["output"])
        except Exception as e:
            print(e)
            print(item["output"])
        # 如果mappings中的value不都在asnwer中，进行空串补全
        for key in dict_list:
            if key not in output_data:
                output_data[key] = "NA"
        new_list.append(
            {"instruction": item["instruction"], "input": item["input"], "output": json.dumps(output_data, ensure_ascii=False)}
        )
    with open(file_name, "w", encoding="utf-8") as outfile:
        outfile.write(json.dumps(new_list, indent=4, ensure_ascii=False))
    return True


if __name__ == "__main__":
    import os

    print(os.getcwd())
    with open('utils/mapping_answer_zh_en.json', 'r', encoding='utf-8') as f:
        mapping = json.load(f)
    with open("data/extract1k_en.json", "r", encoding="utf-8") as f_ori:
        data = json.load(f_ori)
        new_data = insert_loc_in_answer(data, "Treatment Drug Plan", "Is Treatment Drug Recommended", "No")
    with open("data/extract1k_en.json", "w", encoding="utf-8") as f_new:
        json.dump(new_data, f_new, indent=4, ensure_ascii=False)
    # 补全NA deprecated 从开始都加上NA
    # fill_NA_answer("data/extract512_en.json")

    # TODO 单元层级--prompt ?
    # 翻译中文数据集至英文，Google translate
    # translate_zh_dataset("data/extract512_zh_v2.json")

    # 从增强的数据文件生成数据集文件。
    # generate_extract_dataset(
    #     "../lc-medical-record-recognition/data_augmentation/dataset_augmentation_append1.json",
    #     "data/extract512_zh_v2.json",
    # )

    default_unit_locs = {
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
        "治疗用药方案": ["治疗开始日期", "治疗用药名称", "治疗结束日期", "治疗用药是否为建议"],
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
        "日期": ["入院日期", "出院日期", "病史采集日期", "记录日期"],
    }

    # with open('utils/mapping_answer_zh_en.json', 'r', encoding='utf-8') as f:
    #     mapping = json.load(f)

    # with open('nex_dataset/test/exrtract64_test_en.json', 'r', encoding='utf-8') as f:
    #     data = json.load(f)
    # for ds in data:
    #     unit_output = {}
    #     output = json.loads(ds['output'])
    #     # 针对结构进行填充
    #     for unit_name,locs in default_unit_locs.items():
    #         en_unit_name = mapping.get(unit_name)
    #         if en_unit_name not in unit_output:
    #             unit_output[en_unit_name] = {}
    #         for loc in locs:
    #             en_loc = mapping.get(loc)
    #             if not en_loc:
    #                 raise Exception(f'error: {loc}')
    #             unit_output[en_unit_name][en_loc] = output.get(en_loc, 'NA')

    #     ds['output'] = json.dumps(unit_output, ensure_ascii=False)

    # with open("nex_dataset/test/exrtract64_test_en.json", "w", encoding="utf-8") as f:
    #     json.dump(data, f, ensure_ascii=False)
