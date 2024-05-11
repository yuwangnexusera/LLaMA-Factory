import re
from db_unit_dataset import *
from playhouse.shortcuts import (
    model_to_dict,
)
import random
import google_translate
import sys
from pypinyin import Style, pinyin

sys.path.append(".")
from data_augmentation import prompt_dict


def check_zh(string):
    zh_pattern = re.compile("[\u4e00-\u9fa5]")  # 匹配中文字符的正则表达式
    match = zh_pattern.search(string)
    if match:
        start_index = match.start()
        end_index = match.end() - 1  # 结束索引需要减1，因为match.end()返回的是下一个字符的索引
        return True, start_index, end_index
    else:
        return False, None, None


def zh_pinyin(chinese_text):
    pinyin_list = pinyin(chinese_text, style=Style.NORMAL)

    # 拼接拼音
    pinyin_text = "".join([word[0] for word in pinyin_list])

    return pinyin_text


def check_ds_zh(ds_path):
    with open(ds_path, "r", encoding="utf-8") as file:
        dataset_ori = json.load(file)
    for item in dataset_ori:
        output = json.loads(item["output"])
        report = item["input"]
        flag, start_report, end__report = check_zh(report)
        if flag:
            print(f"报告中包含中文:{report[start_report:end__report+1]}")
        for unit_name, locs in output.items():
            for loc in locs:
                for key, value in loc.items():
                    if key == "Diagnosing Doctor":
                        loc[key] = zh_pinyin(value)
                        continue
                    if isinstance(value, list):
                        for v in value:
                            contains_zh, start, end = check_zh(v)
                            if contains_zh:
                                # 调用 prompt_dict.mapping(value) 进行中英文转换
                                translated_value = prompt_dict.mapping_loc_zh_en(v)
                                loc[key] = loc[key].replace(v, translated_value)
                                print(
                                    "在字符串 '{}' 中发现中文，起始索引为{}，结束索引为{}，中文内容为 '{}'，转换后的内容为 '{}'".format(
                                        v, start, end, v[start : end + 1], translated_value
                                    )
                                )
                    else:
                        if isinstance(value, int):
                            loc[key] = f"{value}"
                            continue
                        contains_zh, start, end = check_zh(str(value))
                        if contains_zh:
                            # 调用 prompt_dict.mapping(value) 进行中英文转换
                            translated_value = prompt_dict.mapping_loc_zh_en(value)
                            loc[key] = loc[key].replace(value, translated_value)
                            print(
                                "在字符串 '{}' 中发现中文，起始索引为{}，结束索引为{}，中文内容为 '{}'，转换后的内容为 '{}'".format(
                                    value, start, end, value[start : end + 1], translated_value
                                )
                            )
        item["output"] = json.dumps(output, ensure_ascii=False)
    with open(ds_path, "w", encoding="utf-8") as file:
        json.dump(dataset_ori, file, ensure_ascii=False, indent=4)
    return True


def insert_loc_in_answer(data, unit_name, loc, val):
    for d in data:
        output = json.loads(d["output"])
        for unit, vals in output.items():
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
                "Your task is to extract medical information from the input report and output it in JSON format, and output NA for information not mentioned in the report"
            )
            input = google_translate.translate_text(ds_item["input"])
            if not input:
                print("error input")
            output_zh = json.loads(ds_item["output"])["output"]
            for key, value in output_zh.items():
                if value == "NA":
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
    """去除列表长度大于1，全NA的对象，并且补全对象中key不全的情况"""
    with open(file_name, "r", encoding="utf-8") as file:
        json_data = json.load(file)
    unit_locs = prompt_dict._get_report_structure("出入院记录")
    en_unit_locs = {}
    for unit, locs in unit_locs.items():
        en_unit = prompt_dict.mapping_loc_zh_en(unit)
        en_vals = [prompt_dict.mapping_loc_zh_en(val) for val in locs]
        en_unit_locs[en_unit] = en_vals
    new_list = []
    for item in json_data:
        output_data = json.loads(item["output"])
        cleaned_data = {}  # 存储去除NA的
        for unit, locs in output_data.items():
            if isinstance(locs, list) and len(locs) > 1:
                valid_dicts = []
                for loc_dict in locs:
                    # 检查是否每个值都是'NA'
                    if not all(value == "NA" for value in loc_dict.values()):
                        # 补全缺失的键
                        complete_loc_dict = {key: loc_dict.get(key, "NA") for key in en_unit_locs.get(unit, [])}
                        valid_dicts.append(complete_loc_dict)
                if valid_dicts:
                    cleaned_data[unit] = valid_dicts
            else:
                complete_loc_dict = {key: locs[0].get(key, "NA") for key in en_unit_locs.get(unit, [])}
                cleaned_data[unit] = [complete_loc_dict]
        new_list.append(
            {"instruction": item["instruction"], "input": item["input"], "output": json.dumps(cleaned_data, ensure_ascii=False)}
        )  # 将清理后的数据追加到new_list中
    with open(file_name, "w", encoding="utf-8") as outfile:
        outfile.write(json.dumps(new_list, indent=4, ensure_ascii=False))
    return True


def transfer_output_format(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        for item in data:
            if isinstance(item["output"], str):
                item["output"] = json.loads(item["output"])
            elif isinstance(item["output"], dict):
                item["output"] = json.dumps(item["output"], ensure_ascii=False)
        with open(file_path, "w", encoding="utf-8") as f_new:
            json.dump(data, f_new, indent=4, ensure_ascii=False)


def json_to_jsonl_or_json(input_file_path, output_file_path):
    # Determine input data type based on file extension
    is_jsonl = input_file_path.endswith(".jsonl")

    # Read data from input file
    with open(input_file_path, "r", encoding="utf-8") as input_file:
        if is_jsonl:
            data = [json.loads(line.strip()) for line in input_file]
        else:
            data = json.load(input_file)

    # Convert data to target format
    if is_jsonl:
        # JSONL to JSON
        json_data = []
        for item in data:
            input_report = item[0]["prompt"]
            answer = item[0]["response"][0][0]
            json_data.append(
                {
                    "instruction": "Your task is to extract medical information from the input report and output it in JSON format, and output NA for information not mentioned in the report",
                    "input": input_report,
                    "output": json.loads(answer),
                }
            )
        with open(output_file_path, "w", encoding="utf-8") as output_file:
            output_file.write(json.dumps(json_data, ensure_ascii=False))
    else:
        with open(output_file_path, "w", encoding="utf-8") as output_file:
            for item in data:
                jsonl_data = [{"prompt": item["input"], "response": json.dumps(item["output"], ensure_ascii=False, indent=3)}]
                output_file.write(json.dumps(jsonl_data, ensure_ascii=False) + "\n")
                output_file.flush()


if __name__ == "__main__":
    import os

    print(os.getcwd())
    # json<->jsonl(baidu)
    # json_to_jsonl_or_json( "nex_dataset/train/extract1k_en.jsonl","data/extract1k_en.json")

    # json<->json str
    # transfer_output_format("data/extract1k_en.json")

    # 检查中文，并且走mapping_zh_en
    check_ds_zh("data/extract1k_en.json")

    fill_NA_answer("data/extract1k_en.json")

    # with open('utils/mapping_answer_zh_en.json', 'r', encoding='utf-8') as f:
    #     mapping = json.load(f)
    # with open("data/extract1k_en.json", "r", encoding="utf-8") as f_ori:
    #     data = json.load(f_ori)
    #     new_data = insert_loc_in_answer(data, "Treatment Drug Plan", "Is Treatment Drug Recommended", "No")
    # with open("data/extract1k_en.json", "w", encoding="utf-8") as f_new:
    #     json.dump(new_data, f_new, indent=4, ensure_ascii=False)

    # 补全NA deprecated 从开始都加上NA

    # TODO 单元层级--prompt ?
    # 翻译中文数据集至英文，Google translate
    # translate_zh_dataset("data/extract512_zh_v2.json")

    # 从增强的数据文件生成数据集文件。
    # generate_extract_dataset(
    #     "../lc-medical-record-recognition/data_augmentation/dataset_augmentation_append1.json",
    #     "data/extract512_zh_v2.json",
    # )

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
