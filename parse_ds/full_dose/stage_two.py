# 讲原样提取的内容落到列表选择中
import pandas
import prompt_template
from openai import OpenAI
from ask_model import AskModel
import json
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s-%(levelname)s-%(message)s")


def prepare_stage_two_ds(unit_name, save_path):
    ask_model = AskModel("doubao")
    stage_1_ds = pandas.read_json(save_path, orient="records").to_dict(orient="records")
    stage_two = []
    for report in stage_1_ds:
        # if stage_two.__len__()>4:
        #     break
        try:
            stage_1_output = json.loads(report["output"])
            if isinstance(stage_1_output, dict):
                stage_1_output = [stage_1_output]
        except:
            print(report["input"])
            continue

        for output in stage_1_output:
            if len(stage_1_output) >= 2:
                print()
            # 单个日期的检测内容单独调一次模型？有没有更优的方法
            gene_testing = output.get("基因检测", "")
            if gene_testing not in ["NA", ""]:
                prompt = prompt_template.domain_prompt.get(unit_name).replace("report", gene_testing)
                logging.info(f"{gene_testing}")
                stage_2_output = ask_model.ask_doubao(prompt)
                stage_2_output = stage_2_output.replace("```json", "").replace("```", "")
            else:
                stage_2_output = "[]"
            stage_two.append(
                {
                    "instruction": prompt_template.sft_prompt_stage2.get(unit_name),
                    "input": gene_testing,
                    "output": json.dumps(json.loads(stage_2_output), ensure_ascii=False),
                }
            )
            # 提取 与调用分别修改
    pandas.DataFrame(stage_two).to_json(
        save_path.replace(".json", "_stage_two.json"), orient="records", force_ascii=False, indent=4
    )


if __name__ == "__main__":

    unit_path_map = {
        "基因检测": "data/Genetic Testing/full_dose_ds.json",
        "病理": "data/Pathology/full_dose_ds.json",
        #  "治疗用药方案": "data/Treatment Drug Plan/full_dose_ds.json",
        #  "日期": "data/date_unit/full_dose_ds.json",
        "疾病": "data/Disease/full_dose_ds.json",
        "免疫检测": "data/Immune Testing/full_dose_ds.json",
        "合并疾病": "data/Comorbid Disease/full_dose_ds.json",
    }
    unit_name = "基因检测"
    prepare_stage_two_ds(unit_name, save_path=unit_path_map.get(unit_name))
