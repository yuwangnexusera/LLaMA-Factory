import pandas as pd
import os
'''
1、从recheck_ds整理的SFT数据中获取
2、SFT时调用标注的prompt
3、对比结果是否相同，使用计算F1的方法
4、若相同continue，否则SFT作为rejected，label作为chosen
[
  {
    "instruction": "人类指令（必填）",
    "input": "人类输入（选填）",
    "chosen": "优质回答（必填）",
    "rejected": "劣质回答（必填）"
  }
]

'''


class RLHF_Dataset:
    def __init__(self, sft_path):
        # 打印当前路径
        print(os.getcwd())
        self.sft_path = sft_path
        self.sft_ds = self._get_sft_dataset()
        self.rlhf_ds = []

    def _get_sft_dataset(self):
        if not os.path.exists(self.sft_path):
            raise FileNotFoundError(f"File {self.sft_path} does not exist")
        # pandas读取json文件
        sft_ds = pd.read_json(self.sft_path).to_dict(orient="records")
        return sft_ds
    # 从sft的数据中截取一部分做强化
    def parse_sft_rlhf(self):
        sft_ds = self._get_sft_dataset()
        for sft in sft_ds:
            instruction = sft["instruction"]
            input = sft["input"]
            chosen = sft["output"]
            rejected = ""
            self.rlhf_ds.append({
                "instruction": instruction,
                "input": input,
                "chosen": chosen,
                "rejected": rejected
            }) 

    def save_test(self):
        pd.DataFrame(self.rlhf_ds).to_json(self.sft_path, orient="records", force_ascii=False, lines=False)


if __name__== "__main__":
    rlhf_ds = RLHF_Dataset("data/drug_treatment_plan_recheck_zh.json")
