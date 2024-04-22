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


# 从增强数据到训练数据 test
def generate_extract_dataset(
    aug_data_path,
    train_data_path,
):
    with open(aug_data_path, "r", encoding="utf-8") as file, open(train_data_path, "a", encoding="utf-8") as train_file:
        datasets_json = json.load(file)
        converted_data = []
        for category, datasets in datasets_json.items():
            for dataset in datasets:
                input_val = dataset.get("report", "")
                if not input_val:
                    continue
                dataset.pop("report", None)
                converted_data.append(
                    {
                        "instruction": "你的任务是从输入报告中提取医学信息，并以json格式输出，输入报告：",
                        "input": input_val + "json结果：",
                        "output": json.dumps(dataset, ensure_ascii=False),
                    }
                )
        train_file.write(json.dumps(converted_data, ensure_ascii=False) + "\n")

        train_file.flush()


def translate_zh_dataset(file_name):
    en_dataset = []
    with open(file_name, "r", encoding="utf-8") as zh_file, open(
        file_name.replace("_zh", "_en"), "w", encoding="utf-8"
    ) as en_file:
        data = json.load(zh_file)
        i = 1
        for ds_item in data:
            i = i + 1
            print(len(ds_item["input"]), "----", i)
            instruction = google_translate.translate_text(ds_item["instruction"])
            if not instruction:
                print("error instruction")
            input = google_translate.translate_text(ds_item["input"])
            if not input:
                print("error input")
            output = google_translate.translate_text(ds_item["output"])
            if not output:
                print("error output")
            en_dataset.append({"instruction": instruction, "input": input, "output": output})

        en_file.write(json.dumps(en_dataset, indent=4, ensure_ascii=False))

        en_file.flush()


def write_records_to_jsonl(
    records,
    jsonl_path,
):

    with open(
        jsonl_path,
        "a",
        encoding="utf-8",
    ) as file:
        for item in records:
            category = ss_report_type.select(ss_report_type.report_type).where(ss_report_type.id == item.report_id)[0].report_type
            category_en = google_translate.translate_text(category)

            if not item.content:
                content_obj = hx_ocr_result.select().where(hx_ocr_result.url == item.url)[0]
                content = google_translate.translate_text(content_obj.ocr_result["data"]["ocr_align"])
            else:
                content = google_translate.translate_text(item.content)
            # 假设system_msg是一个你需要定义的变量
            data = {
                "conversations": [
                    {
                        "from": "human",
                        "value": f"Your task is to determine the type of medical report entered. The report type options are: ['Other', 'Examination Record', 'Doctor's Order', 'Prescription', 'Injection', 'Expense', 'Physical Examination Report' ', 'Gene testing', 'Surgery record', 'Examination record', 'Treatment record', 'Pathology report', 'Disease course record', 'Outpatient medical record', 'Discharge and admission record', 'Informed consent', ' Other consultation records', 'Pathological consultation records', 'Other disease diagnosis certificates', 'Outpatient disease diagnosis certificates', 'Inpatient and admission disease diagnosis certificates']\nMedical report: {content}\nOutput format: directly select the most appropriate item from the options and output it, without explaining or outputting redundant content",
                    },
                    {
                        "from": "assistant",
                        "value": f"{category_en}",
                    },
                ]
            }

            # 将data转换为JSON字符串并写入文件，每个对象后面跟一个换行符
            file.write(
                json.dumps(
                    data,
                    ensure_ascii=False,
                )
                + "\n"
            )
            file.flush()
    print("success")

def fill_NA_answer():
    mappings = {
        "出院日期": "Discharge date",
        "入院日期": "Date of admission",
        "ECOG日期": "ECOG Date",
        "ECOG": "ECOG",
        "病理日期": "Pathology date",
        "病理类型": "Pathological type",
        "免疫检测日期": "Immunization test date",
        "TPS": "TPS",
        "PDL1": "PDL1",
        "CPS": "CPS",
        "IC": "IC",
        "诊断医生": "Diagnostic doctor",
        "病史采集日期": "Date of medical history collection",
        "记录日期": "Record date",
        "治疗开始日期": "Treatment start date",
        "治疗结束日期": "Treatment end date",
        "肿瘤具体治疗方式": "Specific tumor treatment",
        "治疗用药名称": "Name of therapeutic drug",
        "手术部位": "Surgical site",
        "脑转移日期": "Date of brain metastasis",
        "脑转部位": "Brain rotation area",
        "基因检测日期": "Date of genetic testing",
        "EGFR": "EGFR",
        "ALK": "ALK",
        "KRAS": "KRAS",
        "BRAF": "BRAF",
        "MET": "MET",
        "RET": "RET",
        "ROS1": "ROS1",
        "NTRK": "NTRK",
        "HER2(ERBB2)": "HER2(ERBB2)",
        "FGFR": "FGFR",
        "BRCA": "BRCA",
        "TP53": "TP53",
        "KEAP1": "KEAP1",
        "STK11": "STK11",
        "HER4（ERBB4）": "HER4 (ERBB4)",
        "RB1": "RB1",
        "HER3（ERBB3）": "HER3 (ERBB3)",
        "疾病首次确诊日期": "Date the disease was first diagnosed",
        "第一次病理确诊时间（穿刺、术后病理等）": "Time of first pathological diagnosis (puncture, postoperative pathology, etc.)",
        "第一次切肺手术时间": "Time of first lung resection surgery",
        "第一次影像确诊时间": "Time of first imaging diagnosis",
        "第一次治疗时间（药物、放疗等）": "Time of first treatment (drugs, radiotherapy, etc.)",
        "首发症状时间": "Time of first symptoms",
        "疾病名称": "Disease Name",
        "出生日期": "date of birth",
        "年龄": "age",
        "性别": "gender",
        "内分泌及免疫系统疾病": "Endocrine and immune system diseases",
        "神经系统疾病": "Nervous system disease",
        "消化系统疾病": "Digestive system diseases",
        "呼吸系统疾病": "Respiratory diseases",
        "循环系统疾病": "Circulatory system diseases",
        "传染性疾病": "Infectious Diseases",
        "恶性肿瘤情况": "Malignant tumor",
        "泌尿生殖系统疾病": "Urogenital system diseases",
        "眼耳鼻喉相关疾病": "Eye, ear, nose and throat related diseases",
    }
    with open('data/extract512_en.json','r',encoding='utf-8') as file:
        json_data = json.load(file)
    new_list = []
    for item in json_data:
        try:
            asnwer = json.loads(item["output"])
        except Exception as e:
            print(e)
            print(item["output"])
        # 如果mappings中的value不都在asnwer中，进行空串补全
        for key,value in mappings.items():
            if value not in asnwer.keys():
                asnwer[value] = 'NA'
        new_list.append({'instruction':item['instruction'],'input':item['input'],'output':json.dumps(asnwer,ensure_ascii=False)})
    with open('data/extract512_en_na.json','w',encoding='utf-8') as outfile:
        outfile.write(json.dumps(new_list,indent=4,ensure_ascii=False))
        outfile.flush()
    return True

if __name__ == "__main__":
    import os
    print(os.getcwd())
    # 补全NA
    # fill_NA_answer()

    # 翻译中文数据集至英文，Google translate
    # translate_zh_dataset("nex_dataset/test/exrtract64_test_zh.json")

    # 从增强的数据文件生成数据集文件。
    # generate_extract_dataset(
    #     "../lc-medical-record-recognition/data_augmentation/dataset_augmentation_test.json",
    #     "nex_dataset/test/dataset_augmentation_test.json",
    # )
    # convert_35_lf('nex_dataset/train/category_train_3_5.jsonl','data/category_zh.json')
    # query_dataset = ss_unit_dataset.select(
    #     ss_unit_dataset.url, ss_unit_dataset.content, ss_unit_dataset.report_id
    # ).group_by(ss_unit_dataset.url)
    # dataset_list = list(query_dataset.execute())
    # records_train, records_validation, records_test = split_dataset(
    #     dataset_list, [0.6, 0.2, 0.2]
    # )
    # test_record_for_gpt = []
    # test_record_for_gpt_url = []
    # for test_record in records_test:
    #     test_record_for_gpt_url.append(test_record.url)
    #     report_type_query = ss_report_type.select(ss_report_type.report_type).where(
    #         ss_report_type.id == test_record.report_id
    #     )[0]
    #     if not test_record.content:
    #         content_obj = hx_ocr_result.select().where(
    #             hx_ocr_result.url == test_record.url
    #         )[0]
    #         test_record.content = content_obj.ocr_result["data"]["ocr_align"]
    #     test_record_for_gpt.append(
    #         {
    #             "url": test_record.url,
    #             "content": google_translate.translate_text(test_record.content),
    #             "report_type": google_translate.translate_text(
    #                 report_type_query.report_type
    #             ),
    #             "before_ft": "",
    #             "after_ft": "",
    #         }
    #     )
    # train_list = [
    #     pic for pic in records_train if pic.url not in test_record_for_gpt_url
    # ]
    # validation_list = [
    #     pic for pic in records_validation if pic.url not in test_record_for_gpt_url
    # ]
    # write_records_to_json(test_record_for_gpt, "nex_dataset/test/category_test_en.json")
    # write_records_to_jsonl(
    #     train_list,
    #     "nex_dataset/train/category_train_en.jsonl",
    # )
    # write_records_to_jsonl(
    #     validation_list,
    #     "nex_dataset/validation/category_validation_en.jsonl",
    # )
