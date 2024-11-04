import re
from google.cloud import translate
import os
from datetime import datetime
# credential_path = "utils\M400806-7257e31dba3d.json"
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credential_path
# Initialize Translation client
def check_zh(string):
    zh_pattern = re.compile("[\u4e00-\u9fa5]")  # 匹配中文字符的正则表达式
    match = zh_pattern.search(string)
    if match:
        start_index = match.start()
        end_index = match.end() - 1  # 结束索引需要减1，因为match.end()返回的是下一个字符的索引
        return True, start_index, end_index
    else:
        return False, None, None


def translate_text(
    text: str = "YOUR_TEXT_TO_TRANSLATE", project_id: str = "agent-400806",source_language_code: str = "zh-CN", target_language_code: str = "en-US",task="trans"
) -> translate.TranslationServiceClient:
    """Translating Text."""
    try:
        client = translate.TranslationServiceClient()

        location = "global"

        parent = f"projects/{project_id}/locations/{location}"

        # Translate text from English to French
        # Detail on supported types can be found here:
        # https://cloud.google.com/translate/docs/supported-formats
        if not text:
            text = "Oh, No"
        check_zh_flag, start_index, end_index = check_zh(text)
        if not check_zh_flag and task != "trans":
            return text
        response = client.translate_text(
            request={
                "parent": parent,
                "contents": [text],
                "mime_type": "text/plain",  # mime types: text/plain, text/html
                "source_language_code": source_language_code,
                "target_language_code": target_language_code,
            }
        )
        print(f"{datetime.now()}-translate:{text}->{response.translations[0].translated_text}")
        return response.translations[0].translated_text
    except Exception as e:
        print("Error:", e)
        return ""


if __name__ == "__main__":
    locs = [
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
    mappings = {}
    for index, loc in enumerate(locs):
        print(index, loc)
        res_loc = translate_text(loc)
        mappings[loc] = res_loc
    print(mappings)
