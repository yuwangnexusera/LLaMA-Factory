from recheck_ds import AlignDataset

# 生成SFT与R:HF数据集总入口
import json

with open("utils/mapping_answer_zh_en.json", "r", encoding="utf-8") as f:
    mapping = json.load(f)
# 治疗用药方案
unit = "病理"
en_unit = mapping.get(unit)
all_path = f"data/{en_unit}/all_zh.json"
sft_path = f"data/{en_unit}/sft_zh.json"
rlhf_path = f"data/{en_unit}/rlhf_zh.json"
test_path = f"data/{en_unit}/test_zh.json"
alignor = AlignDataset(unit,"train")

# 6：4分割all数据集，仅在RLHF时开启
sft,rlhf = alignor.parse_sft_rlhf(all_path)
alignor.save(sft_path,sft)
alignor.save(rlhf_path,rlhf)
# TODO 记得去掉-NA

print("@syu:")
