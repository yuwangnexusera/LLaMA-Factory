import ijson
import json
import pandas as pd
def read_json(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        # 逐个对象解析
        for item in ijson.items(f, "item"):
            data.append(item)
    return data

def parse_json(file_path):
    ori_data = read_json(file_path)
    data = []
    for report in ori_data:
        for question in report:
            data.append({"instruction": "", "input": question[0], "output": json.dumps(question[1], ensure_ascii=False)})
    return data

def write_json(file_path, data):
    pd.DataFrame(data).to_json(file_path, orient='records', force_ascii=False,indent=4)
if __name__ == "__main__":
    ori_file_path = 'data/train_gt.json'
    parsed_data = parse_json(ori_file_path)
    write_json('data/match_zy/train_gt_1.json', parsed_data)
