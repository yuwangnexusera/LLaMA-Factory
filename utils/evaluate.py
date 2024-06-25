import json
def evaluate(path):
    with open(path , "r", encoding="utf-8") as file:
        data = json.load(file)
        test_rate = 0
        right_test = 0
        right_ft = 0
        right_ft_prompt = 0

        for item in data:
            if item["report_type"].strip() == item["llama7b_8bit"].strip():
                right_test += 1

        print(f"llama2 7b FT right answer {right_test}-{len(data)}")


def transfer():
    # 读取jsonl文件
    train_list = []
    with open("nex_dataset/train/category_train_en.jsonl", "r", encoding="utf-8") as file:
        for line in file:
            data = json.loads(line)["conversations"]
            train_list.append({"instruction":f"{data[0]['value']}","input":"","output":f"{data[1]['value']}"})
    with open("nex_dataset/validation/category_validation_en.jsonl", "r", encoding="utf-8") as file:
        for line in file:
            data = json.loads(line)["conversations"]
            train_list.append(
                {
                    "instruction": f"{data[0]['value']}",
                    "input": "",
                    "output": f"{data[1]['value']}",
                }
            )
    with open("data/category_en.json", "w", encoding="utf-8") as file:
        json.dump(train_list, file, ensure_ascii=False)

if __name__ == "__main__":
    # transfer()
    evaluate("nex_dataset/test/llama2_7b_8.json")
