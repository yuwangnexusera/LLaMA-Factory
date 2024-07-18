from openai import OpenAI
import tiktoken
import sys
import json

sys.path.append(".")
from utils import ds_label_wrapper
import pandas
gene_prompt_template = """你是一名出色的基因检测专家，具备出色的专业技能，能够准确解读医学报告中的各种医疗数据。通过临床经验和深厚的医学知识，你能够精准地识别并理解报告中的基因检测信息。注意报告中是以日期为分隔的段落。
请根据下面提供的报告解析出每次基因检测中的有关基因突变的信息。基因检测中可能包含EGFR, ALK, KRAS等基因的突变状态。需要你将每次基因检测的结果创建为一个JSON对象，JSON对象中的键和值的详细描述在下面的'输出格式中'；如果有多次基因检测，就将多个JSON对象放到一个列表中。请确保所有的信息直接来源于报告，并避免任何推理。若有多条不同检查日期的记录，继续以相同格式列出，并放到列表中。
特别注意：
若有多条不同检查日期的记录，继续以相同格式列出，并放到列表中。
请在输出前再次检查一下内容它们非常非常重要：
1. 再检查一遍输出是否正确，如果有多条检查记录，请输出多个json。
2. 请确认你没有漏掉任何基因检测结果。
3. 如果报告中提到“基因”两个字，则“患者是否进行基因检测”选项输出为是。"
  输入报告：|||input_report|||
  输出格式: The output should be a markdown code snippet formatted in the following schema, including the leading and trailing ```json and ```:
[{
 "基因检测日期": string  //输出格式为'%Y-%m-%d',不要推理。
 "患者是否进行基因检测": string  // 选项为['是','否']
 "基因检测": "当你判断出他做了基因检测时,就直接提取基因的名称和突变类型。注意:所有提取的信息必须直接来源于报告文本,不允许基于文本内容进行任何形式的推断、假设或改变。"
}]
"""
gene_sft_prompt = """你是一名出色的基因检测专家，你能够精准地识别并理解报告中的基因检测信息。\
确保所有的信息直接来源于报告，并避免任何推理。若有多条不同检查日期的记录，继续以相同格式列出，并放到列表中。
  输出格式: The output should be a markdown code snippet formatted in the following schema, including the leading and trailing ```json and ```:
[{
 "基因检测日期": string  //输出格式为'%Y-%m-%d',不要推理。
 "患者是否进行基因检测": string  // 选项为['是','否']
 "基因检测": "当你判断出他做了基因检测时,就直接提取基因的名称和突变类型。注意:所有提取的信息必须直接来源于报告文本,不允许基于文本内容进行任何形式的推断、假设或改变。"
}]
  输入报告：
"""
def ask_doubao(prompt):
    client = OpenAI(
        api_key="61ae527f-0de1-4ad9-8f5f-18b8ba296c1f",
        base_url="https://ark.cn-beijing.volces.com/api/v3",
    )

    # Non-streaming:
    print("----- standard request -----")
    completion = client.chat.completions.create(
        model="ep-20240718032206-lj48n",
        messages=[
            {"role": "system", "content": "你的任务是按要求从给定报告中提取医学信息。"},
            {"role": "user", "content": prompt},
        ],
        stream=True,
        temperature=0.01,
    )
    output_llm = ""
    encoding = tiktoken.encoding_for_model(model_name="gpt-4")  # gpt-4/gpt-3.5-turbo
    token_num = len(encoding.encode(prompt))
    for chunk in completion:
        if chunk.choices[0].delta.content is not None:
            print(chunk.choices[0].delta.content, end="")
            output_llm += chunk.choices[0].delta.content
    return output_llm


def gene_prompt_extract():
    prompt_template = gene_prompt_template
    save_path = "../full_dose_ds.json"
    reports = pandas.read_json(save_path, orient="records").to_dict(orient="records")
    exist_urls = [report["url"] for report in reports]
    for img_data in ds_label_wrapper.query_by_nums(507):
        if img_data.preprocess:
            prompt = prompt_template.replace("|||input_report|||", img_data.preprocess)
            # 如果img_data.preprocess不在reports中的report字段,reports是一个json列表，再执行，否则进行下一轮循环
            if img_data.url_desensitive in exist_urls:
                continue
            output = ask_doubao(prompt)
            if "```json" in output:
                output = output.replace("```json", "").replace("```", "")
            try:
                json_output = json.loads(output)
                if len(json_output) > 1:
                    print("json_output is a list")
            except:
                print("json.loads error")
                continue

            reports.append({"url": img_data.url_desensitive, "report": img_data.preprocess, "output": output})
            print(len(reports))

    try:
        pandas.DataFrame(reports).to_json(save_path, orient="records", force_ascii=False, indent=4)
    except Exception as e:
        print(e)
    return reports


def parse_generate_json(save_path="data/Genetic Testing/full_dose_ds.json"):
    gene_full_dose = []
    reports = pandas.read_json("parse_ds/full_dose/gene_500.json", orient="records").to_dict(orient="records")
    for report in reports:
        gene_full_dose.append({"instruction": gene_sft_prompt, "input": report["report"], "output": report["output"]})
    pandas.DataFrame(gene_full_dose).to_json(save_path, orient="records", force_ascii=False, indent=4)
    return gene_full_dose

if __name__ == "__main__":
    full_dose_res = parse_generate_json()
