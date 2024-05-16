import json
import sys
from datetime import datetime
from pypinyin import pinyin, Style

sys.path.append(".")
from utils.google_translate import translate_text


def process_single_loc_obj(dic):
    if isinstance(dic, dict):
        new_dic = {}
        for k, v in dic.items():
            if v == ["NA"]:
                print()
            if k in ["imageUrl", "positions", "general_res", "picId", "入院日期", "出院日期", "病史采集日期", "记录日期"]:
                continue
            if ("日期" in k or "时间" in k ) and isinstance(v,list):
                v = v[0]
            elif k in ["治疗用药名称","诊断医生"]:
                v = split_mul_str_list(v)
            if k == ["NA"]:
                print()
            new_key = mapping_loc_zh_en(k)  # 如果key不存在于maps中，保留原始键
            if new_key:
                if isinstance(v, list):
                    # 只映射列表中存在于maps的值
                    new_dic[new_key] = [mapping_loc_zh_en(v_i) for v_i in v]  # 默认为原始值
                elif "日期" in k or "时间" in k or v in ["NA",["NA"]] or k == "年龄":
                    new_dic[new_key] = v
                else:
                    # 映射值，如果值不存在于maps中，使用原始值
                    new_dic[new_key] = mapping_loc_zh_en(v)
        return new_dic  # 返回新字典代替修改原字典
    elif isinstance(dic, list):
        return [process_single_loc_obj(d) for d in dic]  # 使用列表推导处理列表中的每个元素
    else:
        print(dic)
def split_mul_str_list(v_str):
    '''类似治疗用药可多条，用','隔开转为列表'''
    if ',' in v_str:
        return v_str.split(',')
    elif '|' in v_str:
        return v_str.split('|')
    else:
        return v_str


def mapping_comorbid_disease(comorbid_disease):
    if isinstance(comorbid_disease, dict):
        comorbid_disease = [comorbid_disease]

    new_comorbid_disease = []
    for item in comorbid_disease:
        if "信息来源" not in item:
            continue
        new_cd = {
            "Date of Confirmed Disease": "NA",
            "Information Source": "NA",
            "Infectious Diseases": "NA",
            "Respiratory System Diseases": "NA",
            "Circulatory System Diseases": "NA",
            "Malignant Tumor Conditions": "NA",
            "Digestive System Diseases": "NA",
            "Nervous System Diseases": "NA",
            "Urogenital System Diseases": "NA",
            "Eye, Ear, Nose, and Throat Related Diseases": "NA",
            "Endocrine and Immune System Diseases": "NA",
        }
        new_cd["Information Source"] = mapping_loc_zh_en(item["信息来源"])
        new_cd[mapping_loc_zh_en(item["疾病系统"])] = mapping_loc_zh_en(item["合并疾病"])
        new_cd["Date of Confirmed Disease"] = item["合并疾病确诊日期"]
        new_comorbid_disease.append(new_cd)
    return new_comorbid_disease

if __name__ == "__main__":
    from selenium import webdriver
    from selenium.webdriver.common.keys import Keys
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    import time

    # 启动 Chrome 浏览器
    driver = webdriver.Chrome()

    try:
        # 打开目标网页
        driver.get("https://chatgpt.com/c/3a8f6d61-3b5c-4aad-acb2-58c5ec3d9795")

        # 等待页面加载
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, "prompt-textarea")))

        # 如果需要登录，找到并填写登录表单
        username_field = driver.find_element(By.NAME, "username")
        password_field = driver.find_element(By.NAME, "password")

        # 输入用户名和密码（替换为你的登录信息）
        username_field.send_keys("your_username")
        password_field.send_keys("your_password")

        # 提交登录表单
        password_field.send_keys(Keys.RETURN)

        # 等待登录完成
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, "prompt-textarea")))

        # 找到文本框
        text_area = driver.find_element(By.ID, "prompt-textarea")

        # 发送消息
        message = "Hello, ChatGPT!"
        text_area.send_keys(message)

        # 模拟按下回车键发送消息
        text_area.send_keys(Keys.RETURN)

        # 等待一会儿以观察结果
        time.sleep(5)

    finally:
        # 关闭浏览器
        driver.quit()


# with open("nex_dataset/test/extract_with_unit.json", "r", encoding="utf-8") as f:
#     data = json.load(f)
#     for item_test in data:
#         new_item = {}
#         output = json.loads(item_test["output"])
#         continue
#         # ori_date_unit = output["基本信息"][0]
#         # date_unit = {
#         #     "Discharge Date": ori_date_unit["出院日期"],
#         #     "Admission Date": ori_date_unit["入院日期"],
#         #     "Record Date": ori_date_unit["记录日期"],
#         #     "Medical History Collection Date": ori_date_unit["病史采集日期"],
#         # }
#         # for unit, locs in output.items():
#         #     en_unit = mapping_loc_zh_en(unit)
#         #     if en_unit and en_unit not in new_item:
#         #         new_item[en_unit] = []
#         #     if unit == "合并疾病":
#         #         processed_locs = mapping_comorbid_disease(locs)
#         #     else:
#         #         processed_locs = process_single_loc_obj(locs)
#         #     new_item[en_unit] = processed_locs
#         #     new_item["Date"] = date_unit
#         # item_test["output"] = json.dumps(new_item)

# with open("nex_dataset/test/extract_with_unit.json", "w", encoding="utf-8") as f:

#     json.dump(data, f, ensure_ascii=False)
