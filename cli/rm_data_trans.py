import pandas as pd
import json
import random

if __name__ == "__main__":
    df = pd.read_json("data/gan/test_data_sft.json").to_dict(orient="records")
    random.shuffle(df)
    res = []
    for item in df:
        res.append(
            {
                "conversations": [
                    {
                        "from": "human",
                        "value": f"""你的任务是根据医学实体：{json.dumps(item['units'],ensure_ascii=False)}，生成一份‘肺癌’‘{item['report_type']}’.\
                            医学实体到材料的规则如下：1、医学实体中的键是医学实体类型，值是该实体类型对应的具体值，日期值表示在该日期下做了检查或用药；
                        2、生成的材料中必须包含医学实体中所有非空、非NA的键值"""
                    }
                ],
                "chosen": {"from": "gpt", "value": item["ocr"]},
                "rejected": {"from": "gpt", "value": item["sft_ocr"]},
            }
        )

    with open("data/gan/test_data_rm.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=4)
