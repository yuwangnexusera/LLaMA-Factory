from recheck_ds import AlignDataset

# 生成SFT与R:HF数据集总入口
import json
# from langchain_community.chat_models.openai import ChatOpenAI
with open("utils/mapping_answer_zh_en.json", "r", encoding="utf-8") as f:
    mapping = json.load(f)
# 治疗用药方案,基本信息,疾病,体征数据,诊断,影像学,基因检测,免疫检测,肿瘤治疗,治疗用药方案,合并疾病,   日期
# TODO 将新数据集补充进来，然后把ipynb里面的方法抽取到命令行中
unit = ["病理","治疗用药方案","基本信息","疾病","体征数据","诊断","影像学","基因检测","免疫检测","肿瘤治疗","治疗用药方案","合并疾病"]
for u in unit:
    en_unit = mapping.get(u)
    test_path = f"data/{en_unit}/test_zh.json"
    all_path = f"data/{en_unit}/all_zh.json"
    # sft_path = f"data/{en_unit}/sft_zh.json"
    # dpo_path = f"data/{en_unit}/dpo_zh.json"
    alignor = AlignDataset(u,"test")
    unit_res = alignor.unit_values()
    alignor.save(test_path, unit_res)
# alignor = AlignDataset("治疗用药方案", "train")
# date_res = alignor.unit_values()
# alignor.save(f"data/Treatment Drug Plan/test_zh.json", date_res)
# # 500条做基因检测
# # DPO 6：4分割all数据集，仅在RLHF时开启
# zh_unit = "治疗用药方案"
# en_unit = mapping.get(zh_unit)
# sft_path = f"data/{en_unit}/sft_zh.json"
# dpo_path = f"data/{en_unit}/dpo_zh.json"
# sft,dpo = alignor.parse_sft_rlhf("data/Treatment Drug Plan/all_zh.json")
# alignor.save(sft_path,sft)
# alignor.save(dpo_path, dpo)
# TODO 记得去掉-NA，患者是否进行基因检测 NA->否

print("@syu:")
