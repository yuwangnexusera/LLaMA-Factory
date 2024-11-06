import pandas as pd
import json

if __name__ == "__main__":
    df = pd.read_json("data/gan/test_data_sft.json").to_dict(orient="records")
    res = []
    for item in df:
        res.append(
            {
                "conversations": [
                    {
                        "from": "human",
                        "value": f"你的任务是根据给定的医学实体，生成类型为:{item['report_type']}的报告,报告中必须包含医学实体中所有非空、非NA的值.医学实体：{json.dumps(item['units'],ensure_ascii=False)}",
                    }
                ],
                "chosen": {"from": "gpt", "value": item["ocr"]},
                "rejected": {"from": "gpt", "value": item["sft_ocr"]},
            }
        )

    with open("data/gan/test_data_rm.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=4)
