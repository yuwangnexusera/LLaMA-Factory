import re
import json
from datetime import datetime

import re
import json


def parse_ner_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()

    # 精简并预编译正则表达式模式
    test_i_pattern = re.compile(r"(test_\d+\s*=\s*.*?)\s*#\s*GROUND\s*TRUTH\s*NER\s*DETECTIONS", re.DOTALL)
    test_detections_pattern = re.compile(r"#\s*GROUND\s*TRUTH\s*NER\s*DETECTIONS\s*test_detections\s*=\s*\[(.*?)\]", re.DOTALL)

    # 查找所有 test_i 块
    test_i_blocks = test_i_pattern.findall(content)

    # 查找所有 test_detections 块
    test_detections_blocks = test_detections_pattern.findall(content)

    results = []
    for test_i, test_detections in zip(test_i_blocks, test_detections_blocks):
        try:
            test_i_value = test_i.split("=", 1)[1].strip().strip('"')
            test_i_value = re.sub(r"\s*\n\s*", " ", test_i_value)

            # 处理 test_detections 块
            # 去除末尾可能存在的逗号
            test_detections_clean = test_detections.rstrip(",")

            # 去除换行和空格
            test_detections_clean = test_detections_clean.replace("\n", "").replace(" ", "")

            # 补充缺失的大括号，以保证是一个完整的JSON数组
            test_detections_value = json.loads(f"[{test_detections_clean.strip()}]")
            if not test_detections_value:
                print(f"{test_detections_value}")
                continue
            results.append({"txt": test_i_value, "answer": json.dumps(test_detections_value)})
        except Exception as e:
            print(f"处理错误：{e}")
            continue

    return results

if __name__ == "__main__":
    
    # 使用示例
    print(datetime.now())
    file_path = r"C:\Users\Administrator\Desktop\sfan\test.txt"
    results = parse_ner_file(file_path)
    print(datetime.now())
    with open("data/Sfan/ner_sfan_test.json", "w", encoding="utf-8") as file:
        res = []
        instruction = """Your mission is to extract entity information from biomedical text predictions.
    output format:
        [{   
            "entity_type": ""//list of options：[protein, cell_type, cell_line, DNA, RNA]   
            "entity_value": ""   
            "start_position": ""    
            "end_position": ""
        }]
    biomedical text:"""
        for result in results:
            res.append({"instruction": instruction, "input": result["txt"], "output": result["answer"], "sft_answer": ""})
        file.write(json.dumps(res, ensure_ascii=False, indent=4))
        print("写入成功")
