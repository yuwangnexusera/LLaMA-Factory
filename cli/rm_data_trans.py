import pandas as pd
import json
import random

if __name__ == "__main__":
    df = pd.read_json("data/dpo/test_data_sft.json").to_dict(orient="records")
    random.shuffle(df)
    res = []
    for item in df:
        res.append(
            {
                "conversations": [
                    {
                        "from": "human",
                        "value": f"""你的任务是根据医学实体字典：{json.dumps(item['units'],ensure_ascii=False)}，生成一份‘肺癌’的‘{item['report_type']}’.\
                            从医学实体到{item['report_type']}的生成规则如下：\
                                1、医学实体字典中的键是医学实体类型，值是该实体类型对应的具体值；\
                                2、生成的材料中必须包含医学实体中所有的值\
                                3、值为NA或者为空的键值忽略""",
                    }
                ],
                "chosen": {"from": "gpt", "value": item["ocr"]},
                "rejected": {"from": "gpt", "value": item["sft_ocr"]},
            }
        )

    with open("data/dpo/test_data_rm.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=4)
