from db_unit_dataset import *
from playhouse.shortcuts import model_to_dict
import random
import google_translate

def split_dataset(dataset_list, ratios):

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

    return first_part, second_part, third_part


# 打开文件准备写入
def generate_other(jsonl_path, records_80_percent):
    with open(jsonl_path, "a", encoding="utf-8") as file:
        for item in records_80_percent:

            category = (
                ss_report_type.select(ss_report_type.report_type)
                .where(ss_report_type.id == item.report_id)[0]
                .report_type
            )
            content = item.content  # 假设每一项都有.content属性
            if not content:
                content_obj = hx_ocr_result.select().where(
                    hx_ocr_result.url == item.url
                )[0]
                content = content_obj.ocr_result["data"]["ocr_align"]
            # 假设system_msg是一个你需要定义的变量
            train_validation_data = {
                "conversations": [
                    {
                        "from": "human",
                        "value": f"Your task is to determine the type of medical report entered. The report type options are: ['Other', 'Examination Record', 'Doctor's Order', 'Prescription', 'Injection', 'Expense', 'Physical Examination Report' ', 'Gene testing', 'Surgery record', 'Examination record', 'Treatment record', 'Pathology report', 'Disease course record', 'Outpatient medical record', 'Discharge and admission record', 'Informed consent', ' Other consultation records', 'Pathological consultation records', 'Other disease diagnosis certificates', 'Outpatient disease diagnosis certificates', 'Inpatient and admission disease diagnosis certificates']\nMedical report: {content}\nOutput format: directly select the most appropriate item from the options and output it, without explaining or outputting redundant content",
                    },
                    {"from": "assistant", "value": f"{category}"},
                ]
            }
            file.write(json.dumps(train_validation_data, ensure_ascii=False) + "\n")
            file.flush()

    print(jsonl_path)


def write_records_to_json(records, file_name):
    with open(file_name, "w", encoding="utf-8") as file:
        json.dump(records, file, ensure_ascii=False, indent=4)
        file.flush()


def write_records_to_jsonl(records, jsonl_path):

    with open(jsonl_path, "a", encoding="utf-8") as file:
        for item in records:
            category = (
                ss_report_type.select(ss_report_type.report_type)
                .where(ss_report_type.id == item.report_id)[0]
                .report_type
            )
            category_en = google_translate.translate_text(category)

            if not item.content:
                content_obj = hx_ocr_result.select().where(
                    hx_ocr_result.url == item.url
                )[0]
                content = google_translate.translate_text(
                    content_obj.ocr_result["data"]["ocr_align"]
                )
            else:
                content = google_translate.translate_text(item.content)
            # 假设system_msg是一个你需要定义的变量
            data = {
                "conversations": [
                    {
                        "from": "human",
                        "value": f"Your task is to determine the type of medical report entered. The report type options are: ['Other', 'Examination Record', 'Doctor's Order', 'Prescription', 'Injection', 'Expense', 'Physical Examination Report' ', 'Gene testing', 'Surgery record', 'Examination record', 'Treatment record', 'Pathology report', 'Disease course record', 'Outpatient medical record', 'Discharge and admission record', 'Informed consent', ' Other consultation records', 'Pathological consultation records', 'Other disease diagnosis certificates', 'Outpatient disease diagnosis certificates', 'Inpatient and admission disease diagnosis certificates']\nMedical report: {content}\nOutput format: directly select the most appropriate item from the options and output it, without explaining or outputting redundant content",
                    },
                    {"from": "assistant", "value": f"{category_en}"},
                ]
            }

            # 将data转换为JSON字符串并写入文件，每个对象后面跟一个换行符
            file.write(json.dumps(data, ensure_ascii=False) + "\n")
            file.flush()
    print("success")


if __name__ == "__main__":
    query_dataset = ss_unit_dataset.select(
        ss_unit_dataset.url, ss_unit_dataset.content, ss_unit_dataset.report_id
    ).group_by(ss_unit_dataset.url)
    dataset_list = list(query_dataset.execute())
    records_train, records_validation, records_test = split_dataset(
        dataset_list, [0.6, 0.2, 0.2]
    )
    test_record_for_gpt = []
    test_record_for_gpt_url = []
    for test_record in records_test:
        test_record_for_gpt_url.append(test_record.url)
        report_type_query = ss_report_type.select(ss_report_type.report_type).where(
            ss_report_type.id == test_record.report_id
        )[0]
        if not test_record.content:
            content_obj = hx_ocr_result.select().where(
                hx_ocr_result.url == test_record.url
            )[0]
            test_record.content = content_obj.ocr_result["data"]["ocr_align"]
        test_record_for_gpt.append(
            {
                "url": test_record.url,
                "content": google_translate.translate_text(test_record.content),
                "report_type": google_translate.translate_text(
                    report_type_query.report_type
                ),
                "before_ft": "",
                "after_ft": "",
            }
        )
    train_list = [
        pic for pic in records_train if pic.url not in test_record_for_gpt_url
    ]
    validation_list = [
        pic for pic in records_validation if pic.url not in test_record_for_gpt_url
    ]
    write_records_to_json(test_record_for_gpt, "nex_dataset/test/category_test_en.json")
    write_records_to_jsonl(
        train_list,
        "nex_dataset/train/category_train_en.jsonl",
    )
    write_records_to_jsonl(
        validation_list,
        "nex_dataset/validation/category_validation_en.jsonl",
    )
