import json
import os
import random
import re

# from langchain_community.chat_models import ChatOpenAI

from langchain_openai import ChatOpenAI
import os
import sys
import warnings

from dotenv import load_dotenv
import numpy as np

sys.path.append(".")
from utils import conf

from utils.TokenAndCost import TokenCalculate

import prompt_dict

# from report import *
from utils.logger import logger

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

os.environ["OPENAI_API_KEY"] = conf.OPENAI_API_KEY
# os.environ["OPENAI_API_KEY"] = "sk-SaF2Q6QYGo1INej98x1vM0itqQrvC16v3J5ae3LDL9HnMAPH"


def ask_llm_return_str(prompt, model_name):
    llm_openai = ChatOpenAI(model=model_name, openai_api_base=conf.OPENAI_API_BASE, temperature=0.9, streaming=True)
    try:
        logger.info(f"{model_name}：start chain.run")
        output = llm_openai.invoke(prompt)
        logger.info("end chain.run")
    except Exception as e:
        logger.error(f"llm_openai.invoke 异常{e}")
    return output.content


def clean_json_string(s):
    """使用正则表达式移除字符串中所有的非法控制字符"""
    s = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", s)
    return s


def generate_report_structure(report_type, model_name="gpt-3.5-turbo-free"):
    template_report = f"""请根据医院医疗报告的常见标准格式，告诉我肺癌患者{report_type}应包含哪些部分，每个部分应该含有哪些信息，请不要输出任何说明性文字。\
    """
    report_struct = ask_llm_return_str(template_report, model_name=model_name)
    # print(f"@@@@{report_type}:{report_struct}")
    return report_struct


def generate_report(prompt_data, report_type, report_structure, model_name="gpt-3.5-turbo-free"):
    template = f"""你作为一名肺癌领域的肿瘤科临床医生，任务是根据下面提供的‘医学数据’按照‘{report_type}’的结构编写一份{report_type}。\
    请避免罗列数据,把日期置于相关检查之前，以自然语言形式描述,500-1000个汉字，直接输出生成的报告。\
    {report_type}的结构如下：{report_structure}\
    ##医学数据：{prompt_data}##"""
    generate_res = ask_llm_return_str(template, model_name=model_name)
    # print("@@@@initial_report:{}".format(generate_res))

    return generate_res


def insert_loc_to_report(ori_report, locs, model_name):
    template = f"""作为一名肺癌领域的肿瘤科临床医生，我需要你辅助我完成数据增强任务。将下面提供的‘##医学数据##’插入到下述‘##出入院记录##’的合理位置，并按要求输出新的‘出入院记录’。\
    要求：1.保证医学数据全部插入到新输出的‘出入院记录’中的对应位置，符合一份出入院记录的整体结构。比如基因检测的数据请放在报告中检测结果内容之后，个人信息请放在报告开头部分。\
        2.插入后对文本进行部分修改，你可以将提供的‘出入院记录’原句做同义转换或某些医学名词做同义词转换，保证上下文表述的连贯。请直接输出新的‘出入院记录’。\
    ‘出入院记录’：{ori_report}\
    ##医学数据：{locs}##"""
    generate_res = ask_llm_return_str(template, model_name=model_name)
    # print("@@@@initial_report:{}".format(generate_res))

    return generate_res


def generate_feedback(initial_report, report_type, model_name="gpt-3.5-turbo-free"):
    template_feedback = f"""你的任务是阅读并分析我编写的这份肺部肿瘤科的{report_type},直接分点给出不足之处或者需要改善的地方,直接输出修改建议。\
        从{report_type}的结构完整性、信息准确性、文字表述的逻辑性与连贯性等方面考虑。着重注意:1、避免数据简单罗列,在编写医学报告时，确保各项给定的医学数据准确对应其相应的医学检查类别，避免将特定检查的数据错误地归类到其他检查项目中。\
            2、给出修改建议后，填充的数据务必是真实的,不必过于纠结日期问题。\
    {report_type}：{initial_report}
    """

    report_feedback = ask_llm_return_str(template_feedback, model_name)
    # print("@@@@feedback:{}".format(feedback))
    return report_feedback


def generate_refine(feedback, report_type, initial_report, model_name="gpt-3.5-turbo-free"):
    template_refine = f"""你的任务是根据下面提供的 "修改建议" 优化这份肺癌肿瘤科的{report_type},必须保证医学数据的内容不能改变,直接输出优化完的报告。\
    {report_type}:{initial_report}
    修改建议：{feedback}
    """
    report_refine = ask_llm_return_str(template_refine, model_name)
    # print("@@@@report_refine:{}".format(report_refine))
    return report_refine


def generate_recheck(report, report_type, prompt_data, model_name):
    template_recheck = f"""作为一名临床医生，你的任务是优化我提供的{report_type}，使其足够完整且包含更丰富的信息,以下要求中的各项都必须满足,尤其是1,2两点。要求：\
        1、确保‘医疗数据’中的值都在‘{report_type}’中,如果有缺失的医疗数据,将缺失的医疗数据补充在{report_type}的合适位置。\
        2、将报告中'待补充','XXX','待填充'等非确定性数据根据上下文补充完整。\
        3、在给定报告的基础上进行文字的扩充和优化，使其更符合医学报告自然语言的描述。\
        4、确保报告中的数据、诊断、治疗建议严格基于准确的医学证据和检查结果，并验证所有数据或结果均正确归类至相应的检查项目中。同时，检查报告是否包括所有必要信息，并确保信息按逻辑顺序组织，如排列不合理，则需按照确保医学数据内容不变的原则重新调整语言表述。\
    医疗数据:{json.dumps(prompt_data, ensure_ascii=False).replace("{", "").replace("}", "")}
    {report_type}：{report}\
        请务必不要加上推理或者描述性语句，直接输出优化完的医学报告。
    """
    checked_report = ask_llm_return_str(template_recheck, model_name)
    # print("@@@@checked_report:{}".format(checked_report))
    return checked_report


def insert_mul_en_loc(loc, report):

    prompt = f""""Your task is to insert medical data in the appropriate location of the medical report as required. \
        Requirements:\
            1. First, analyze where the input report describes the {list(loc.keys())}. \
            2. According to the location analysis of the first point, the following Json format medical data are converted into professional descriptions of clinical medical reports and placed in the analyzed location,\
            3. Ensure that the insert data without changing other medical information in the report, convert medical data into a context-appropriate description, ensure that all medical information appears in the report! \n
        medical data :{loc}.\n
        Medical report: {report} \n
        Output format: directly output the modified medical report without any explanatory notes."""
    model_name = random.choices(
        [
            "gpt-4-turbo",
            "ERNIE-Bot 4.0",
            "gpt-3.5-turbo",
            "Llama-2-70b-chat",
            "Baichuan2-53B",
        ],
        [0.4, 0.2, 0.3, 0.05, 0.05],
        k=1,
    )[0]
    tokenize = TokenCalculate(model_name)
    new_report = ask_llm_return_str(prompt, model_name)
    logger.info(f"input:{tokenize.token_count(prompt)},output:{tokenize.token_count(report)}")
    return new_report


def generate_dataset_by_answer(report_type):
    flag = False
    # 随机取值
    prompt_data = prompt_dict.generate_domain_data(report_type)
    # model_name  ["ERNIE-Bot 4.0", "gpt-3.5-turbo-1106", "gpt-4-turbo", "gpt-3.5-turbo-free"]
    if report_type == "出入院记录":
        # 出入院记录拆开生成
        report_type = random.choice(["出院记录", "入院记录", "出入院记录"])
    # 报告结构
    report_structure = generate_report_structure(report_type, model_name="gpt-3.5-turbo-free")
    # 生成报告
    initial_report = generate_report(prompt_data, report_type, report_structure, model_name="gpt-3.5-turbo-0125")

    # mult self refine
    report_feedback = ""
    report_refine = ""
    for i in range(2):
        try:
            if initial_report:
                # LLM feedback
                report_feedback = generate_feedback(initial_report, report_type, model_name="gpt-3.5-turbo-free")
            if report_feedback:
                # self refine
                report_refine = generate_refine(report_feedback, report_type, initial_report, model_name="gpt-3.5-turbo-1106")
            initial_report = report_refine
            flag = True
        except Exception as e:
            logger.error("generate_dataset_by_answer 异常{}".format(e))
            flag = False
            break
    if flag:
        initial_report = generate_recheck(initial_report, report_type, prompt_data, model_name="gpt-4-turbo")
        prompt_data["report"] = initial_report
        return prompt_data
    else:
        return None


def calculate_avg_token(report_list):
    lengths = []
    token_calculate = TokenCalculate("gpt-3.5-turbo-1106")

    for report in report_list:
        if report["report"] != "":
            token_length = token_calculate.token_count(report["report"])
            lengths.append(token_length)
        else:
            logger.warning("report=''")

    if lengths:
        # 定义区间，例如从 800 到 1500，每100个为一个区间
        bins = np.arange(0, 1200, 100)

        # 计算各个区间的数量
        histogram, bin_edges = np.histogram(lengths, bins)

        logger.warning(f"最大token长度为: {np.max(lengths)}")
        logger.warning(f"中位数token长度为: {np.median(lengths)}")
        logger.warning(f"平均token长度为: {np.mean(lengths)}")

        # 打印各个区间的数量
        for i in range(len(bin_edges) - 1):
            print(f"长度区间 {bin_edges[i]}-{bin_edges[i+1]}报告数: {histogram[i]} ")
    else:
        logger.warning("没有有效的报告长度数据可供分析。")


def remove_duplicate():
    file_path = "data_augmentation/dataset_augmentation_append1.json"

    # 读取 JSON 数据
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    # 使用集合来跟踪唯一的 input 值
    seen_inputs = set()
    new_data = []

    for item in data:
        input_value = item.get("input")
        if input_value not in seen_inputs:
            seen_inputs.add(input_value)
            new_data.append(item)

    # 将处理后的数据写回文件
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(new_data, file, indent=4, ensure_ascii=False)

    print(f"重复项已移除并已更新文件。{len(data)-len(new_data)}")
    return


#   插入英文点位到报告中 可多条的
def insert_en_loc(dataset_path, test_data_path):
    # with open(test_data_path, "r", encoding="utf-8") as f:
    #     test_data = json.load(f)

    unit_to_insert = [
        # "Treatment Drug Plan",
        "Cancer treatment"
    ]
    # 检查哪些单元经常出现可多条
    # for item in test_data:
    #     output = json.loads(item["output"])
    #     for unit_name,vals in output.items():
    #         if isinstance(vals,list) and len(vals)>1:
    #             unit_to_insert.add(unit_name)
    index_set = set()
    num = 1
    for i in range(100):
        print(num)
        num += 1
        if num > 40:
            break
        with open(dataset_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        index = random.randint(0, len(dataset) - 1)
        if index in index_set:
            logger.info(f"index:{index},pass--")
            continue
        index_set.add(index)

        # 获取选中的数据项
        item_data = dataset[index]
        output = json.loads(item_data["output"])
        report = item_data["input"]
        unit_selected = random.sample(unit_to_insert, random.randint(1, len(unit_to_insert)))
        unit_domain = {}
        for unit in unit_selected:
            if len(output[unit]) > 1:
                logger.info(f"index:{index},unit-{unit}-length={len(output[unit])},pass--")
                continue
            single_unit_domain = prompt_dict.generate_domain_unit_en("出入院记录", unit)
            unit_domain[unit] = single_unit_domain
            output[unit].append(single_unit_domain)
        if unit_domain == {}:  # or unit_domain["Treatment Drug Plan"]["Treatment Drug Names"] == []
            logger.info(f"null")
            continue
        new_report = insert_mul_en_loc(unit_domain, report)
        dataset[index]["input"] = new_report + "json output:"
        dataset[index]["output"] = json.dumps(output, ensure_ascii=False)
        with open(dataset_path, "w", encoding="utf-8") as outfile:
            json.dump(dataset, outfile, ensure_ascii=False, indent=4)
            logger.info(f"写入完成{index}")
    return


def insert_loc_in_report(file_name):
    # 出入院的key列表
    all_keys = [
        "出院日期",
        "入院日期",
        "ECOG日期",
        "ECOG",
        "病理日期",
        "病理类型",
        "免疫检测日期",
        "TPS",
        "PDL1",
        "CPS",
        "IC",
        "诊断医生",
        "病史采集日期",
        "记录日期",
        "治疗开始日期",
        "治疗结束日期",
        "肿瘤具体治疗方式",
        "治疗用药名称",
        "手术部位",
        "脑转移日期",
        "脑转部位",
        "基因检测日期",
        "EGFR",
        "ALK",
        "KRAS",
        "BRAF",
        "MET",
        "RET",
        "ROS1",
        "NTRK",
        "HER2(ERBB2)",
        "FGFR",
        "BRCA",
        "TP53",
        "KEAP1",
        "STK11",
        "HER4（ERBB4）",
        "RB1",
        "HER3（ERBB3）",
        "疾病首次确诊日期",
        "第一次病理确诊时间（穿刺、术后病理等）",
        "第一次切肺手术时间",
        "第一次影像确诊时间",
        "第一次治疗时间（药物、放疗等）",
        "首发症状时间",
        "疾病名称",
        "出生日期",
        "年龄",
        "性别",
        "内分泌及免疫系统疾病",
        "神经系统疾病",
        "消化系统疾病",
        "呼吸系统疾病",
        "循环系统疾病",
        "传染性疾病",
        "恶性肿瘤情况",
        "泌尿生殖系统疾病",
        "眼耳鼻喉相关疾病",
    ]
    # 根据已有报告插入点位进行增强,# TODO 选几个点位，加入到报告的合适位置
    with open("data_augmentation/1.jsonl", "r", encoding="utf-8") as file:
        lines = file.readlines()
    random.shuffle(lines)
    # 遍历文件的每一行
    i = 0
    for line in lines:
        i += 1
        if i > 50:
            break
        new_answer = {}
        json_obj = json.loads(line)[0]
        prompt = json_obj["prompt"]
        # 截取prompt字符串中'医学报告:'至'标注:'之间的内容，冒号可能是中文的也可能是英文的
        match = re.search(r"医学报告[：:]?\s*(.*?)(?:\s*标注[：:]?\s*|$)", prompt, re.DOTALL)
        if match:
            report = match.group(1)  # 返回匹配到的内容
        else:
            print("report匹配失败")
            continue
        response_list = json_obj["response"][0]
        try:
            answer = json.loads(response_list[0])
            # 去掉单元层级
            for key, value in answer.items():
                if isinstance(value, list):
                    value = value[0]
                # 各单元的值
                for k, v in value.items():
                    if k in all_keys:
                        new_answer[k] = v
            # 补全NA
            for loc in all_keys:
                if loc not in new_answer:
                    new_answer[loc] = "NA"
            # 从new_answer中挑NA的值，随机选择一个value，并且插入报告中
            na_keys = [key for key, value in new_answer.items() if value == "NA"]

            # 从na_keys中随机选择3到7个键，注意需要处理na_keys长度小于3的情况
            num_to_select = random.randint(8, min(20, len(na_keys)))
            selected_keys = random.sample(na_keys, num_to_select)
            value_to_insert = {}
            for key in selected_keys:
                val = prompt_dict.get_locVal("出入院记录", key)
                if not val:
                    continue
                value_to_insert[key] = val
                new_answer[key] = val
            #
            model_name = random.choices(["gpt-3.5-turbo", "gpt-4-turbo", "ERNIE-Bot-turbo"], [0.6, 0.2, 0.2], k=1)[0]
            print(f"model_name:{model_name}")
            _reports = []
            with open(file_name, "r", encoding="utf-8") as file:
                _reports = json.load(file)
            logger.info(f"当前文件中共有{len(_reports)}条")
            new_report = insert_loc_to_report(report, value_to_insert, model_name)
            _reports.append({"input": new_report, "output": new_answer})
            with open(file_name, "w", encoding="utf-8") as f:
                json.dump(_reports, f, ensure_ascii=False, indent=4)
        except Exception as e:
            print(e)

    return True


if __name__ == "__main__":

    # insert_en_loc("data/extract1k_en.json", "nex_dataset/test/extract_with_unit.json")
    print()
    # 加入单元层级
    structure = {
        "日期": {"出院日期": "", "入院日期": "", "病史采集日期": "", "记录日期": ""},
        "基本信息": {"出生日期": "", "年龄": "", "性别": ""},
        "疾病": {
            "疾病首次确诊日期": "",
            "第一次病理确诊时间（穿刺、术后病理等）": "",
            "第一次切肺手术时间": "",
            "第一次影像确诊时间": "",
            "第一次治疗时间（药物、放疗等）": "",
            "首发症状时间": "",
            "疾病名称": "",
        },
        "体征数据": {"ECOG": "", "ECOG日期": ""},
        "诊断": {"诊断医生": ""},
        "影像学": {"脑转移日期": "", "脑转部位": ""},
        "病理": {"病理日期": "", "病理类型": ""},
        "基因检测": {
            "ALK": "",
            "MET": "",
            "RB1": "",
            "RET": "",
            "BRAF": "",
            "BRCA": "",
            "EGFR": "",
            "FGFR": "",
            "KRAS": "",
            "NTRK": "",
            "ROS1": "",
            "TP53": "",
            "KEAP1": "",
            "STK11": "",
            "HER2(ERBB2)": "",
            "HER3（ERBB3）": "",
            "HER4（ERBB4）": "",
            "基因检测日期": "",
        },
        "免疫检测": {"IC": "", "CPS": "", "TPS": "", "PDL1": "", "免疫检测日期": ""},
        "肿瘤治疗": {"手术部位": "", "治疗开始日期": "", "治疗用药名称": "", "治疗结束日期": "", "肿瘤具体治疗方式": ""},
        "治疗用药方案": {"治疗开始日期": "", "治疗用药名称": "", "治疗结束日期": ""},
        "合并疾病": {
            "合并疾病确诊日期": "",
            "信息来源": "",
            "疾病名称": "",
            "传染性疾病": "",
            "呼吸系统疾病": "",
            "循环系统疾病": "",
            "恶性肿瘤情况": "",
            "消化系统疾病": "",
            "神经系统疾病": "",
            "运动系统疾病": "",
            "泌尿生殖系统疾病": "",
            "眼耳鼻喉相关疾病": "",
            "内分泌及免疫系统疾病": "",
        },
    }

    # 根据已有报告插入点位进行增强
    # insert_loc_in_report("data_augmentation/dataset_augmentation_append1.json")
    # remove_duplicate()
    # 查看多少个可多条的比例
    with open('data/extract1k_en.json','r',encoding='utf-8') as f:
        data = json.load(f)
        num = 0
        for d in data:
            for unit,vals in json.loads(d['output']).items():
                if unit == "Cancer treatment":
                    if len(vals) > 1:
                        num+=1
        print(num)
    # 文件路径
    file_path = "data_augmentation/dataset_augmentation_append1.json"
    report_types = ["出入院记录"]
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            existing_data = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        existing_data = {}
    error_data = []
    for category in report_types:
        logger.warning(f"当前‘{category}’共 {len(existing_data.get(category, []))} 条数据")
        # 计算平均token数
        avg_token = calculate_avg_token(existing_data.get(category, []))
        # 每个报告类型生成的报告数
        for i in range(600):
            # new_data = generate_noise_report(existing_data[i])
            try:
                new_data = generate_dataset_by_answer(category)
                if not new_data:
                    continue
            except Exception as e:
                logger.error("generate_dataset_by_answer 异常{}".format(e))
                continue

            if category in existing_data:
                existing_data[category].append(new_data)
            else:
                existing_data[category] = [new_data]
            # 每条写入
            with open(file_path, "w", encoding="utf-8") as file:
                json.dump(existing_data, file, ensure_ascii=False, indent=4)
                logger.warning(f"当前‘{category}’共 {len(existing_data[category])} 条数据")
                file.flush()
