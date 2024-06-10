import json
from random import randint, sample
import random
import re
from attr import attributes
from faker import Faker
import datetime
import sys
sys.path.append(".")
from peewee import fn
from utils.db2 import ss_turmo_medicine
from utils.google_translate import translate_text
fake = Faker(locale="zh_CN")


_default_unit_locs = {
    "基本信息": ["出生日期", "年龄", "性别"],
    "疾病": [
        "疾病首次确诊日期",
        "第一次病理确诊时间（穿刺、术后病理等）",
        "第一次切肺手术时间",
        "第一次影像确诊时间",
        "第一次治疗时间（药物、放疗等）",
        "首发症状时间",
        "疾病名称",
    ],
    "体征数据": ["ECOG", "ECOG日期"],
    "诊断": ["诊断医生"],
    "影像学": ["脑转移日期", "脑转部位"],
    "病理": ["病理日期", "病理类型"],
    "基因检测": [
        "ALK",
        "MET",
        "RB1",
        "RET",
        "BRAF",
        "BRCA",
        "EGFR",
        "FGFR",
        "KRAS",
        "NTRK",
        "ROS1",
        "TP53",
        "KEAP1",
        "STK11",
        "HER2(ERBB2)",
        "HER3（ERBB3）",
        "HER4（ERBB4）",
        "基因检测日期",
        "患者是否进行基因检测"
    ],
    "免疫检测": ["IC", "CPS", "TPS", "PDL1", "免疫检测日期"],
    "肿瘤治疗": ["手术部位", "治疗开始日期", "治疗用药名称", "治疗结束日期", "肿瘤具体治疗方式"],
    "治疗用药方案": ["治疗开始日期", "治疗用药名称", "治疗结束日期", "治疗用药是否为建议"],
    "合并疾病": [
        "合并疾病确诊日期",
        "信息来源",
        "合并疾病",
        # "传染性疾病",
        # "呼吸系统疾病",
        # "循环系统疾病",
        # "恶性肿瘤情况",
        # "消化系统疾病",
        # "神经系统疾病",
        # "泌尿生殖系统疾病",
        # "眼耳鼻喉相关疾病",
        # "内分泌及免疫系统疾病",
    ],
    "日期": ["入院日期", "出院日期", "病史采集日期", "记录日期","就诊日期","检查日期","报告日期","审核日期","送检日期","收到日期","申请日期","会诊日期"],
}
# 报告包含哪些特定点位
_category_locs = {
    "出入院记录": {
        "日期": ["入院日期", "出院日期", "病史采集日期", "记录日期"],
    }
}
# 值域配置区,某种报告在domain_list中，直接调用随机选择，不在domain_list去loc_des找规则
_domain_list = {
    "病理类型": [
        "腺癌",
        "鳞屑样腺癌",
        "腺泡样腺癌",
        "乳头状腺癌",
        "微乳头状腺癌",
        "实性腺癌",
        "浸润性黏液腺癌",
        "混合性浸润性黏液性和非黏液性腺癌",
        "胶体样腺癌",
        "胎儿型腺癌",
        "肠型腺癌",
        "微浸润性腺癌",
        "原位腺癌",
        "黏液表皮样癌",
        "腺样囊性癌",
        "唾液腺型肿瘤",
        "上皮肌上皮癌",
        "鳞癌",
        "角化型鳞状细胞癌",
        "非角化型鳞状细胞癌",
        "基底样鳞状细胞癌",
        "原位鳞状细胞癌",
        "大细胞癌",
        "巨细胞癌",
        "小细胞癌",
        "复合型小细胞癌",
        "腺鳞癌",
        "侵袭前病变",
        "弥漫性特发性肺神经内分泌细胞增生",
        "肉瘤样癌",
        "多形性癌",
        "梭形细胞癌",
        "癌肉瘤",
        "肺母细胞瘤",
        "淋巴上皮瘤样癌",
        "NUT癌",
        "其他病理类型的非小细胞肺癌",
        "胸膜间皮瘤",
        "胸腺癌",
        "混合型非小细胞肺癌",
        "神经内分泌癌",
    ],
    "疾病名称": [
        "非小细胞肺癌",
        "小细胞肺癌",
        "胸膜间皮瘤",
        "胸腺癌",
        "腺癌",
        "鳞屑样腺癌",
        "腺泡样腺癌",
        "乳头状腺癌",
        "微乳头状腺癌",
        "实性腺癌",
        "浸润性黏液腺癌",
        "混合性浸润性黏液性和非黏液性腺癌",
        "胶体样腺癌",
        "胎儿型腺癌",
        "肠型腺癌",
        "微浸润性腺癌",
        "原位腺癌",
        "黏液表皮样癌",
        "腺样囊性癌",
        "唾液腺型肿瘤",
        "上皮肌上皮癌",
        "鳞癌",
        "角化型鳞状细胞癌",
        "非角化型鳞状细胞癌",
        "基底样鳞状细胞癌",
        "原位鳞状细胞癌",
        "大细胞癌",
        "巨细胞癌",
        "小细胞癌",
        "复合型小细胞癌",
        "腺鳞癌",
        "侵袭前病变",
        "弥漫性特发性肺神经内分泌细胞增生",
        "肉瘤样癌",
        "多形性癌",
        "梭形细胞癌",
        "癌肉瘤",
        "肺母细胞瘤",
        "淋巴上皮瘤样癌",
        "NUT癌",
        "其他病理类型的非小细胞肺癌",
        "胸膜间皮瘤",
        "胸腺癌",
        "混合型非小细胞肺癌",
        "神经内分泌癌",
    ],
    "肿瘤具体治疗方式": [
        "手术",
        "消融",
        "胸腔灌注",
        "心包灌注",
        "粒子植入",
        "介入治疗",
        "放疗",
        "同步放化疗",
        "化疗",
        "靶向",
        "免疫",
        "抗血管",
        "内分泌",
        "细胞疗法",
        "器官移植",
        "干细胞移植",
    ],
    "脑转部位": [
        "脑继发恶性肿瘤",
        "大脑半球转移",
        "丘脑转移",
        "小脑转移",
        "脑干转移",
        "脑膜转移",
        "未知",
        "脑转移",
        "无脑转",
    ],
    "手术部位": ["切肺", "脑转移", "肝转移", "其他"],
    "内分泌及免疫系统疾病": [
        "糖尿病",
        "I型糖尿病",
        "痛风",
        "类风湿关节炎",
        "银屑病",
        "系统性红斑狼疮",
        "特应/异皮炎（湿疹）",
        "荨麻疹",
        "干燥综合征",
        "甲状旁腺功能减退症",
        "甲状旁腺功能亢进症",
        "痒疹",
        "重症肌无力",
        "多发性硬化症",
        "狼疮肾炎",
        "白癜风",
        "川崎病",
        "溃疡性结肠炎",
        "克罗恩病",
        "葡萄膜炎",
        "美尼尔病",
        "重症肌无力",
        "格林-巴利综合征",
        "自身免疫性疾病",
    ],
    "神经系统疾病": [
        "阿尔茨海默病",
        "痴呆",
        "癫痫",
        "帕金森",
        "脑梗",
        "炎症",
        "感染",
        "脊髓压迫",
        "精神疾病",
        "周围神经炎",
        "抑郁症",
        "双向情感障碍",
        "精神分裂",
    ],
    "消化系统疾病": [
        "腹泻",
        "出血",
        "溃疡",
        "肝硬化",
        "炎症",
        "结核",
        "穿孔",
        "食管胃底静脉曲张",
        "梗阻",
        "食管裂孔疝",
        "腹水",
        "脂肪肝",
        "消化道结石",
        "消化道造瘘",
        "鼻饲",
    ],
    "呼吸系统疾病": [
        "肺炎",
        "肺感染",
        "肺不张",
        "肺水肿",
        "气胸",
        "血胸",
        "脓胸",
        "肺大疱",
        "间质性肺炎",
        "肺纤维化",
        "肺气肿",
        "阻塞性肺气肿",
        "慢性支气管炎",
    ],
    "循环系统疾病": [
        "高血压",
        "高血脂",
        "冠心病",
        "严重心律失常",
        "心功能不全",
        "心力衰竭",
        "卒中",
        "肝性脑病",
        "管腔狭窄",
        "动脉斑块",
        "炎症",
        "出血",
        "栓塞（肺栓塞）",
        "动脉粥样硬化",
        "埂塞",
        "上腔静脉压塞综合征",
        "肺源性心脏病",
        "肿瘤破裂（血管包绕）",
        "血栓",
    ],
    "传染性疾病": ["甲型肝炎", "乙型肝炎", "丙型肝炎", "艾滋病", "人乳头状瘤病毒感染", "梅毒螺旋体感染", "结核", "癌栓"],
    "恶性肿瘤情况": [
        "癌栓",
        "乳腺癌",
        "结直肠癌",
        "胃癌",
        "肝癌",
        "甲状腺癌",
        "白血病",
        "非霍奇金淋巴瘤",
        "霍奇金淋巴瘤",
        "鼻咽癌",
        "食管癌",
        "胰腺癌",
        "宫颈癌",
        "子宫内膜癌",
        "卵巢癌",
        "肾癌",
        "膀胱癌 ",
    ],
    "泌尿生殖系统疾病": [
        "尿毒症",
        "严重肾功能不全",
        "肾功能衰竭",
        "肾性贫血",
        "高尿酸血症",
        "肾积水",
        "泌尿生殖系统炎症",
        "泌尿生殖系统出血",
        "高血压肾病",
        "肾囊肿",
    ],
    "眼耳鼻喉相关疾病": [
        "青光眼",
        "白内障",
        "干眼症",
        "眼结石",
        "黄斑水肿",
        "黄斑变性",
        "感染",
        "穿孔",
        "软化",
        "出血",
        "溃烂",
        "视网膜脱落",
        "结膜炎",
        "虹膜炎",
    ],
    "血栓":["上肢浅静脉血栓", "上肢深静脉血栓", "上肢动脉血栓", "颅内静脉血栓", "颅内动脉血栓", "下肢浅静脉血栓", "下肢深静脉血栓", "下肢动脉血栓", "内脏器官深静脉血栓", "内脏器官浅静脉血栓", "内脏器官动脉血栓", "弥散性血管内凝血", "血栓性血小板减少性紫癜", "动脉硬化闭塞症", "急性冠状动脉综合征", "硬脑膜动脉瘘", "子痫", "癌性血栓性静脉炎", "肢体/脏器缺血性改变", "溃疡改变"],
    "癌栓":["门静脉主干癌栓", "门静脉左侧分支癌栓", "门静脉右侧分支癌栓", "癌栓累及肠系膜下腔静脉", "癌栓累及肠系膜上腔静脉", "心脏受累", "肝静脉主干癌栓", "肝静脉分支癌栓", "胆管癌栓", "下腔静脉内", "膈肌以上下腔静脉内"],
    "EGFR": [
        "扩增",
        "18阳性",
        "19阳性",
        "20阳性",
        "21阳性",
        "19del",
        "19del/L858R",
        "L858R",
        "T790M",
        "20插入",
        "C797S",
        "G719X",
        "G719A",
        "G719D",
        "G719S",
        "G719C",
        "18del",
        "19ins",
        "S768I",
        "L861Q",
        "E709V",
        "其他罕见突变",
        "阳性",
    ],
    "ALK": ["融合", "点突变", "重排", "插入", "扩增", "易位", "阳性"],
    "KRAS": ["扩增", "G13C", "G12A", "G12C", "G12X", "G213X", "Q61X", "G12D", "G12V", "G13D", "Q61L", "其他罕见突变", "阳性"],
    "BRAF": ["V600E", "非V600", "阳性"],
    "MET": ["MET14跳突", "MET扩增", "c-MET过表达", "c-MET扩增", "其他罕见突变", "MET14插入", "MET14缺失", "MET融合", "阳性"],
    "RET": ["融合", "点突变", "重排", "插入", "缺失", "阳性"],
    "ROS1": ["融合", "点突变", "重排", "G3032R", "S1986F", "S1986Y", "阳性"],
    "HER2(ERBB2)": ["20插入", "非20插入", "阳性"],
    "FGFR": ["FGFR1", "FGFR2", "FGFR3", "FGFR4", "点突变", "融合", "重排", "易位", "阳性"],
    "BRCA": ["BRCA1", "BRCA2", "阳性"],
    "TP53": ["阳性"],
    "KEAP1": ["阳性"],
    "STK11": ["阳性"],
    "HER4（ERBB4）": ["阳性"],
    "RB1": ["阳性"],
    "HER3（ERBB3）": ["阳性"],
    "NTRK": ["NTRK1", "NTRK2", "NTRK3", "阳性", "融合", "点突变", "重排"],
}


def mapping_loc_zh_en(key, trans=True):
    with open("utils/mapping_answer_zh_en.json", "r", encoding="utf-8") as f:
        mapping = json.load(f)
    return_key = mapping.get(key)
    if return_key:
        return return_key
    elif key in ["NA"] or re.match(r"\d{4}-\d{2}-\d{2}", key):
        return key
    elif key not in mapping.keys():
        # 找到相等的value返回key值
        for k, v in mapping.items():
            if v == key:
                return k
        return translate_text(key,task="mapping")


def _get_report_structure(report_type):
    # 优先看_category_locs的unit, 若无则引用_default_unit_locs
    category_unique_locations = _category_locs.get(report_type)
    if category_unique_locations:
        category_unique_locations.update(_default_unit_locs)
    return category_unique_locations


def __generate_date(start_year=2018, start_month=1, end_year=2024,end_month=12):
    """随机生成一个日期"""
    start_date = datetime.datetime(start_year, start_month, 1)
    end_date = datetime.datetime(end_year, end_month, 30)
    days_between_dates = (end_date - start_date).days
    random_number_of_days = randint(0, days_between_dates)
    return (start_date + datetime.timedelta(days=random_number_of_days)).strftime("%Y-%m-%d")


def random_select(key):
    """通过点位名随机从domain列表中选择值"""
    # TODO 加上只能选择一个的key
    # 随机决定选择的元素数量，大部分情况下选择1个
    num_to_select = 1 if random.random() < 0.8 else 2  # 80%的概率选择1个，20%的概率选择2-3个
    selected = random.sample(_domain_list[key], min(num_to_select, len(_domain_list[key])))
    return selected


# 治疗用药方案从数据库取几个药
def __select_drugs():
    # 从ss_turmo_medicine数据库随机取几个药品
    random_drug_records = ss_turmo_medicine.select().order_by(fn.Rand()).limit(7)
    drug_set = set()
    for record in random_drug_records:
        drug_set.add(record.component_name)
        # 取record.component_synonym 第一个顿号、之前的元素
        drug_set.add(record.component_synonym.split(",")[0].split("、")[0])
        # drug_set.add(record.general_name.replace("（未上市）", ""))
        if record.product_name != "nan":
            drug_set.add(record.product_name)
    selected_drugs = random.sample(list(drug_set), fake.random_int(min=0, max=4))
    return selected_drugs


def get_loc_des():
    return {
        "性别": fake.random_element(elements=("男", "女", "未知", "")),
        "IC": fake.random_element(elements=("0", "1", "2", "3")),
        "PDL1": str(fake.random_int(min=1, max=100)) + "%",
        "CPS": fake.random_int(min=1, max=100),
        "年龄": fake.random_element(elements=(fake.random_int(min=35, max=65), "")),
        "ECOG": fake.random_element(elements=("0", "1", "2", "3", "4", "5")),
        "TPS": str(fake.random_int(min=1, max=100)) + "%",
        "诊断医生": fake.name(),
        "治疗用药名称": __select_drugs(),
        # "治疗用药名称": random.sample(["pemigatinib", "carboplatin", "bevacizumab", "anlotinib"],fake.random_int(min=1, max=4)),
        "信息来源": fake.random_element(elements=("既往史", "出院诊断", "诊断", "入院诊断")),
    }


def _get_date_des():
    return {
        "出院日期": __generate_date(start_year=2022, end_year=2023),
        "入院日期": __generate_date(start_year=2019, end_year=2020),
        "检查日期": __generate_date(start_year=2020, end_year=2021),
        "记录日期": __generate_date(start_year=2020, end_year=2021),
        "报告日期": __generate_date(start_year=2020, end_year=2021),
        "审核日期": __generate_date(start_year=2020, end_year=2021),
        "送检日期": __generate_date(start_year=2020, end_year=2021),
        "收到日期": __generate_date(start_year=2020, end_year=2021),
        "申请日期": __generate_date(start_year=2020, end_year=2021),
        "治疗开始日期": __generate_date(start_year=2020, end_year=2021),
        "治疗结束日期": __generate_date(start_year=2021, end_year=2022),
        "ECOG日期": __generate_date(start_year=2021, end_year=2021),
        "基因检测日期": __generate_date(start_year=2021, end_year=2021),
        "免疫检测日期": __generate_date(start_year=2021, end_year=2021),
        "病理日期": __generate_date(start_year=2021, end_year=2021),
        "病史采集日期": __generate_date(start_year=2021, end_year=2021),
        "首次确诊日期": __generate_date(start_year=2020, end_year=2021),
        "治疗开始日期": __generate_date(start_year=2020,start_month=6, end_year=2020,end_month=9),
        "治疗结束日期": __generate_date(start_year=2021,start_month = 1, end_year=2021,end_month=3),
        "疾病首次确诊日期": __generate_date(start_year=2020, end_year=2021),
        "第一次病理确诊时间（穿刺、术后病理等）": __generate_date(start_year=2019, end_year=2020),
        "第一次切肺手术时间": __generate_date(start_year=2021, end_year=2021),
        "第一次影像确诊时间": __generate_date(start_year=2020, end_year=2020),
        "第一次治疗时间（药物、放疗等）": __generate_date(start_year=2020, end_year=2021),
        "首发症状时间": __generate_date(start_year=2021, end_year=2022),
        "脑转移日期": __generate_date(start_year=2021, end_year=2022),
        "出生日期": __generate_date(start_year=1950, end_year=2000),
        "合并疾病确诊日期": __generate_date(start_year=2021, end_year=2022),
        "就诊日期": __generate_date(start_year=2021, end_year=2022),
    }


def generate_domain_data(report_type):
    # 报告类型-单元-点位
    category_unit_location_mapping = _get_report_structure(report_type)
    selection_criteria = {
        "基因检测": 4,
        "免疫检测": 2,
        "合并疾病":3,
        "治疗用药方案": 3,
        "影像学": 3,
        "肿瘤治疗":3
    }
    # 从相应的报告类型中取出一些unit_name
    selected_data = {}
    for unit_name, locations in category_unit_location_mapping.items():
        if unit_name not in selected_data:
            selected_data[unit_name] = {}
        if locations:
            # 获取当前键的选择个数，如果未指定，则随机生成
            x = selection_criteria.get(unit_name, random.randint(1, len(locations)))
            # 限制 x 不超过列表长度
            x = min(x, len(locations))
            # 从locations中随机选择x个location
            selected_locs = random.sample(locations, x)
            for loc in selected_locs:
                if loc in _domain_list.keys():
                    # 从值域列表中随机选择
                    selected_data[unit_name][loc] = random_select(loc)
                elif "日期" in loc or "时间" in loc:
                    selected_data[unit_name][loc] = _get_date_des()[loc]
                else:
                    selected_data[unit_name][loc] = get_loc_des()[loc]
    # 保证疾病和病理类型一定存在
    if len(selected_data["疾病"].keys()) >= 1 and "疾病名称" not in selected_data["疾病"].keys():
        selected_data["疾病"]["疾病名称"] = random_select("疾病名称")
    if len(selected_data["病理"].keys()) >= 1 and "病理类型" not in selected_data["病理"].keys():
        selected_data["病理"]["病理类型"] = random_select("病理类型")
    return selected_data

def generate_domain_data_en(report_type):
    zh_domain = generate_domain_data(report_type)


def generate_domain_unit_en(report_type,unit_name):
    zh_unit_name = mapping_loc_zh_en(unit_name)
    zh_domain = generate_domain_data(report_type).get(zh_unit_name, [])
    en_unit_domain = {}
    for k_zh,v_zh in zh_domain.items():
        if isinstance(v_zh,list):
            en_unit_domain[mapping_loc_zh_en(k_zh)] = []
            for i_zh in v_zh:
                en_unit_domain[mapping_loc_zh_en(k_zh)].append(mapping_loc_zh_en(i_zh))
        else:
            en_unit_domain[mapping_loc_zh_en(k_zh)] = mapping_loc_zh_en(v_zh)
    return en_unit_domain


def generate_domain_unit_zh(report_type, unit_name):
    res = generate_domain_data(report_type).get(unit_name, [])
    return res


def get_locVal(report_type, loc):
    category_unit_location_mapping = _get_report_structure(report_type)
    if loc not in category_unit_location_mapping:
        return None
    if loc in _domain_list.keys():
        return random_select(loc)
    if loc in _get_date_des().keys():
        return _get_date_des()[loc]


# 不同报告类型包含的keys
if __name__ == "__main__":
    # res = _get_report_structure("出入院记录")
    # select_val = generate_domain_data("出入院记录")
    # print(select_val)
    while(True):
        print(__select_drugs())
