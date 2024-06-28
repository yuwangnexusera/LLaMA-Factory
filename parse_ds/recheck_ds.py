import sys
import os

sys.path.append(".")
from data_augmentation import prompt_dict
import logging
import pandas as pd
import json
from utils import ds_label_wrapper
from sft_prompt import sft_unit_prompt
import random

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# default-locations包含不绑定报告类型日期的其他点位，包含报告类型日期的将在后续@wy TODO 补充
class AlignDataset:
    """
    读取数据库，取 native_result_custom 中的数据,补全所有点位(空的填入NA)，填入方式prompt_dict,数据分成SFT和RLHF，test数据集单独处理
    """

    def __init__(self, unit_name, stage="train Or test"):
        self.unit_name = unit_name
        self.stage = stage
        self.ds = []  # 数据集中所有单元的数据
        self.unit_ds = []  # 数据集中所有单元的数据
        self.load_ds()

    def load_ds(self):
        if self.stage == "train":
            ds_objs = ds_label_wrapper.select_sft_train_ds()
        elif self.stage == "test":
            ds_objs = ds_label_wrapper.select_test()
        else:
            raise ValueError("Unknown stage.")
        """加载所有训练用的数据（ds2000开头）"""
        for ds in ds_objs:
            if ds.native_result_custom:
                self.ds.append({"ocr": ds.ocr_result, "result": ds.native_result_custom})

    def remove_duplicates(self, json_list):

        # 将每个JSON对象转换为字符串，并使用集合去重
        unique_items = {json.dumps(item, sort_keys=True) for item in json_list}
        # 将去重后的字符串转换回JSON对象
        unique_json_list = [json.loads(item) for item in unique_items]
        return unique_json_list

    def fill_na_by_locs(self, unit_data):
        # 对单个对象内进行填充
        ret_dict = {}
        unit_locs = prompt_dict._default_unit_locs.get(self.unit_name, [])
        for loc in unit_locs:
            if loc in unit_data:
                ret_dict[loc] = unit_data[loc]
            else:
                ret_dict[loc] = "NA"
        return ret_dict

    # TODO 去除同一组中重复的数据
    def parse_sft_rlhf(self, all_path):
        # 读取all_path的json列表数据
        with open(all_path, "r", encoding="utf-8") as f:
            unit_json = json.load(f)

        # 打乱数据顺序
        random.shuffle(unit_json)

        # 计算60%和40%的数量
        total_len = len(unit_json)
        sft_len = int(total_len * 0)

        # 分割数据集
        sft_data = unit_json[:sft_len]
        dpo_data = unit_json[sft_len:]

        # 准备返回的SFT数据和RLHF数据
        sft_return = []
        dpo_return = []

        # 处理SFT数据
        for sft in sft_data:
            instruction = sft["instruction"]
            input_data = sft["input"]
            chosen = sft["output"]
            sft_return.append({"instruction": instruction, "input": input_data, "output": chosen})

        # 处理RLHF数据
        for dpo in dpo_data:
            instruction = dpo["instruction"]
            input_data = dpo["input"]
            chosen = dpo["output"]
            rejected = ""  # 如果有实际的rejected数据可以在这里填入
            dpo_return.append(
                {
                    "conversations": [{"from": "human", "value": instruction + input_data}],
                    "chosen": {"from": "gpt", "value": chosen},
                    "rejected": {"from": "gpt", "value": rejected},
                }
            )
        return sft_return, dpo_return

    def unit_values(self):
        # 存入 _all数据集

        for data in self.ds:
            unit_ds = []
            output = data["result"]
            target_unit_data = output.get(self.unit_name)
            if target_unit_data is None:
                continue
            if isinstance(target_unit_data, dict):
                target_unit_data = [target_unit_data]
            elif target_unit_data == [""] or target_unit_data == [] or target_unit_data == "":
                target_unit_data = [{}]
            for index, unit_data in enumerate(target_unit_data):
                res = self.fill_na_by_locs(unit_data)
                # 单元答案[{}]
                unit_ds.append(res)
            self.unit_ds.append(
                {
                    "instruction": sft_unit_prompt.get(self.unit_name, ""),
                    "input": data["ocr"],
                    "output": json.dumps(unit_ds, ensure_ascii=False), #self.remove_duplicates(unit_ds),是否需要全去重？wy觉得没必要
                }
            )
        return self.unit_ds

    def save(self, path, sft_unit_ds):
        """保存到json文件"""
        # 获取目录路径
        directory = os.path.dirname(path)
        # 如果目录不存在，则创建目录
        if not os.path.exists(directory):
            os.makedirs(directory)
        # 将数据保存到JSON文件
        pd.DataFrame(sft_unit_ds).to_json(path, orient="records", force_ascii=False, lines=False, indent=4)
        logging.info(f"{len(sft_unit_ds)},{self.unit_name} saved to {path}.")

    # TODO 报告的信息组
    def dates_info(self):
        date_info_list = []
        date_keys = prompt_dict._default_unit_locs.get("日期")
        all_data = self.ds
        for d_obj in all_data:
            date_dict = {}
            native_custom = d_obj["result"]
            for unit,loc_list in native_custom.items():
                for sin_native_custom in loc_list:
                    if sin_native_custom=="":
                        sin_native_custom = {}
                    for d_k in date_keys:
                        if d_k not in date_dict or (date_dict[d_k] =="NA" and date_dict[d_k]):
                            date_dict[d_k] = sin_native_custom.get(d_k,"NA")
            date_info_list.append({"instruction": sft_unit_prompt.get("日期", ""), "input": d_obj["ocr"], "output": json.dumps(date_dict, ensure_ascii=False)})
        return date_info_list

if __name__ == "__main__":

    unit = "病理"
    alignor = AlignDataset("","train")
    alignor.dates_info()
