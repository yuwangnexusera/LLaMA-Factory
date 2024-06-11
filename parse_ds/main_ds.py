from recheck_ds import AlignDataset
from rlhf import RLHF_Dataset
# 生成SFT与R:HF数据集总入口
import json
with open('utils/mapping_answer_zh_en.json', 'r', encoding='utf-8') as f:
    mapping = json.load(f)
# 治疗用药方案
unit = "病理"
en_unit = mapping.get(unit)
sft_path = f"data/{en_unit}_sft_zh.json"
rlhf_path = sft_path.replace('sft', 'rlhf')
alignor = AlignDataset(unit, 1000, sft_path)
# TODO 记得去掉-NA
res = alignor.extract_specified_values()
alignor.save()
# 测试集的更新
# alignor.extract_test()
# alignor.save_test()
# rlhf
# rlhf = RLHF_Dataset(rlhf_path)
print("@syu:")
