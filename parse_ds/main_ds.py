from recheck_ds import AlignDataset

# 生成SFT与R:HF数据集总入口
import json

with open("utils/mapping_answer_zh_en.json", "r", encoding="utf-8") as f:
    mapping = json.load(f)
# 治疗用药方案,基本信息,疾病,体征数据,诊断,影像学,基因检测,免疫检测,肿瘤治疗,治疗用药方案,合并疾病,   日期
# TODO 单独处理日期
unit = ["病理","治疗用药方案","基本信息","疾病","体征数据","诊断","影像学","基因检测","免疫检测","肿瘤治疗","治疗用药方案","合并疾病"]
for u in unit:
    en_unit = mapping.get(u)
    test_path = f"data/{en_unit}/test_zh.json"
    all_path = f"data/{en_unit}/all_zh.json"
    sft_path = f"data/{en_unit}/sft_zh.json"
    dpo_path = f"data/{en_unit}/dpo_zh.json"
    alignor = AlignDataset(u,"train")
    unit_res = alignor.unit_values()   
    alignor.save(all_path, unit_res)
# 报告日期
# alignor = AlignDataset("", "test")
# date_res = alignor.dates_info()
# alignor.save(f"data/date_unit/test_zh.json", date_res)
# 500条做基因检测
# 6：4分割all数据集，仅在RLHF时开启
# sft,dpo = alignor.parse_sft_rlhf(all_path)
# alignor.save(sft_path,sft)
# alignor.save(dpo_path, dpo)
# TODO 记得去掉-NA，患者是否进行基因检测 NA->否

print("@syu:")
