import re
import json


def parse_ner_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()

    # Regular expressions to match test_i and test_detections blocks
    test_i_pattern = re.compile(r"(test_\d+=.*?)# GROUND TRUTH NER DETECTIONS", re.DOTALL)
    test_detections_pattern = re.compile(r"# GROUND TRUTH NER DETECTIONS\s*test_detections\s*=\s*\[(.*?)\]", re.DOTALL)

    # Find all test_i blocks
    test_i_blocks = test_i_pattern.findall(content)

    # Find all test_detections blocks
    test_detections_blocks = test_detections_pattern.findall(content)

    # Combine the results
    results = []
    for test_i, test_detections in zip(test_i_blocks, test_detections_blocks):
        test_i_value = test_i.split("=", 1)[1].strip()
        test_i_value = re.sub(r"\s*\n\s*", " ", test_i_value).strip().strip('"')
        test_detections_value = f"[{test_detections.strip()}]"

        # Ensure that there is no trailing comma in test_detections_value
        test_detections_value = json.loads(re.sub(r",\s*([\]}])", r"\1", test_detections_value))

        results.append({"txt": str(test_i_value), "answer": json.dumps(test_detections_value)})

    return results


# 使用示例
file_path = "./train.txt"
results = parse_ner_file(file_path)

for result in results:
    print("test_i:", result["test_i"])
    print("test_detections:", result["test_detections"])
    print("-----")
