import sys
sys.path.append(".")
from data_augmentation import prompt_dict
from logging import Logger
import pandas as pd
import json
from utils import ds_label_wrapper 
from sft_prompt import sft_unit_prompt
# default-locations包含不绑定报告类型日期的其他点位，包含报告类型日期的将在后续@wy TODO 补充
class AlignDataset:
    '''
        读取数据库，取 native_result_custom 中的数据,补全所有点位(空的填入NA)，填入方式prompt_dict
    '''
    def __init__(self, unit_name, ds_num, save_path):
        self.unit_name = unit_name
        self.ds_num = ds_num
        self.save_path = save_path
        self.ds = []
        self.unit_ds = []
        self.test_ds = []
        self.test_unit_ds = []
        self.load_ds()
        self.load_test_ds()
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
    def load_ds(self):
        ds_objs = ds_label_wrapper.select_sft_train_ds()
        for ds in ds_objs:
            if ds.native_result_custom:
                self.ds.append({"ocr": ds.ocr_result, "result": ds.native_result_custom})
                # if len(self.ds)>=self.ds_num:
                #     break

    def extract_specified_values(self):

        extracted_values = []

        for data in self.ds:
            output = data["result"]
            target_unit_data = output.get(self.unit_name, None)
            if target_unit_data is not None:
                if isinstance(target_unit_data, dict):
                    target_unit_data = [target_unit_data]
            else:
                continue
            for index, unit_data in enumerate(target_unit_data):
                res = self.fill_na_by_locs(unit_data)
                self.unit_ds.append({"instruction":sft_unit_prompt.get(self.unit_name,""),"input": data["ocr"], "output": json.dumps(res, ensure_ascii=False)})
        return self.unit_ds

    def load_test_ds(self):
        ds_objs = ds_label_wrapper.select_test()
        for ds in ds_objs:
            if ds.native_result_custom:
                self.test_ds.append({"ocr": ds.ocr_result, "result": ds.native_result_custom})

    def extract_test(self):

        extracted_values = []

        for data in self.test_ds:
            output = data["result"]
            target_unit_data = output.get(self.unit_name, None)
            if target_unit_data is not None:
                if isinstance(target_unit_data, dict):
                    target_unit_data = [target_unit_data]
            else:
                continue
            for index, unit_data in enumerate(target_unit_data):
                res = self.fill_na_by_locs(unit_data)
                self.test_unit_ds.append(
                    {
                        "instruction": sft_unit_prompt.get(self.unit_name, ""),
                        "input": data["ocr"],
                        "output": json.dumps(res, ensure_ascii=False),
                    }
                )
        return self.test_unit_ds

    def save(self):
        pd.DataFrame(self.unit_ds).to_json(self.save_path, orient="records",force_ascii=False, lines=False)
    def save_test(self):
        pd.DataFrame(self.test_unit_ds).to_json(self.save_path.replace("sft","test"), orient="records",force_ascii=False, lines=False)
if __name__=='__main__':

    unit = "治疗用药方案"
    en_unit = "drug_treatment_plan"
    alignor = AlignDataset(unit, 1, f"data/{en_unit}_sft_zh.json")
    # res = alignor.extract_specified_values()
    # alignor.save()
    # 测试集
    alignor.extract_test()
    alignor.save_test()
    print("@syu:")
