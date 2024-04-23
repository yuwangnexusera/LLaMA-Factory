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
            output_zh = json.loads(ds_item["output"])
            for key, value in output_zh.items():
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

    # 补全NA
    fill_NA_answer("nex_dataset/test/exrtract64_test_en.json")

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
