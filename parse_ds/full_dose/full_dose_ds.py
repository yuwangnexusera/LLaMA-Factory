from openai import OpenAI
import tiktoken
import sys
import json
import time
sys.path.append(".")
from env.env_llm import OPENAI
from utils import ds_label_wrapper
from parse_ds.full_dose import prompt_template
import pandas

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
    print("answer by :doubao")
    for chunk in completion:
        if chunk.choices[0].delta.content is not None:
            print(chunk.choices[0].delta.content, end="")
            output_llm += chunk.choices[0].delta.content
    return output_llm


def ask_llm(prompt):

    client = OpenAI(
        # This is the default and can be omitted
        api_key=OPENAI["OPENAI_API_KEY"],
        base_url=OPENAI["OPENAI_API_BASE"],
    )
    model_name = "gpt-4o-mini"
    try:
        # 创建聊天完成请求
        completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=model_name,
            temperature=0.01,
            stream=True,
            
            # timeout=120
        )
        # breakpoint()
        print(f"answer by :{model_name}")
        output = ""
        for chunk in completion:
            if chunk.choices[0].delta.content is not None:
                print(chunk.choices[0].delta.content, end="")
                output += chunk.choices[0].delta.content
        # output = completion.choices[0].message.content
    except Exception as e:
        print(f"@ask_llm :{e}")
        output = ""  # 设置为 None，表示提取失败
    return output


def gene_prompt_extract():
    save_path = "../full_dose_ds.json"
    reports = pandas.read_json(save_path, orient="records").to_dict(orient="records")
    exist_urls = [report["url"] for report in reports]
    for img_data in ds_label_wrapper.query_by_nums(507):
        if img_data.preprocess:
            prompt = prompt_template.full_dose_prompt.get("基因检测").replace("|||input_report|||", img_data.preprocess)
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

def conbine_disease_extract():
    conbine_disease_list = []
    ds_500 = pandas.read_json("parse_ds/full_dose/gene_500.json", orient="records").to_dict(orient="records")
    max_retries = 3  # 最大重试次数
    for ds in ds_500:
        retry_count = 0
        prompt = prompt_template.full_dose_prompt.get("合并疾病").replace("|||input_report|||", ds["report"])
        output = ask_doubao(prompt)

        while not output and retry_count < max_retries:
            retry_count += 1
            print(f"No output received for input: {ds['report']}. Retrying... Attempt {retry_count}/{max_retries}")
            time.sleep(10) 
            output = ask_doubao(prompt)

        if not output:
            print(f"Failed to get output after {max_retries} attempts for input: {ds['report']}. Skipping.")
            continue
        conbine_disease_list.append({"instruction": prompt_template.sft_prompt.get("合并疾病"), "input": ds["report"], "output": output})
    pandas.DataFrame(conbine_disease_list).to_json("data/Comorbid Disease/full_dose_ds.json", orient="records", force_ascii=False, indent=4)
    return conbine_disease_list
def parse_generate_json(save_path="data/Genetic Testing/full_dose_ds.json"):
    gene_full_dose = []
    reports = pandas.read_json("parse_ds/full_dose/gene_500.json", orient="records").to_dict(orient="records")
    for report in reports:
        gene_full_dose.append({"instruction": prompt_template.sft_prompt.get("基因检测"), "input": report["report"], "output": report["output"]})
    pandas.DataFrame(gene_full_dose).to_json(save_path, orient="records", force_ascii=False, indent=4)
    return gene_full_dose

if __name__ == "__main__":
    full_dose_res = conbine_disease_extract()
