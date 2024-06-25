sft_unit_prompt = {
    "体征数据": """你的任务是从输入报告中提取'体征数据'的相关信息，原报告中没有明确提到的信息输出'NA',按照下列格式输出：
        [{
            "ECOG日期": ''
            "ECOG": 可选项为['0','1','2','3','4','5','NA']
        }]
        输入报告：
    """,
    "病理": """你的任务是从输入报告中识别'病理'检查的相关信息。如果报告中有多次病理检查，继续以输出格式要求输出到列表中，只关注明确提及的病理检测，不关注基于影像学诊断的结果，无法按照要求提取的信息输出NA。
        输出格式：[{
        "病理日期": ''
        "病理类型": 选项为["腺癌", "鳞屑样腺癌", "腺泡样腺癌", "乳头状腺癌", "微乳头状腺癌", "实性腺癌", "浸润性黏液腺癌", "混合性浸润性黏液性和非黏液性腺癌", "胶体样腺癌", "胎儿型腺癌", "肠型腺癌", "微浸润性腺癌", "原位腺癌", "粘液表皮样癌", "腺样囊性癌", "唾液腺型肿瘤", "上皮肌上皮癌", "浸润性腺癌", "肺腺癌", "鳞癌", "角化型鳞状细胞癌", "非角化型鳞状细胞癌", "基底样鳞状细胞癌", "原位鳞状细胞癌", "大细胞癌", "巨细胞癌", "小细胞癌", "小细胞肺癌", "复合型小细胞癌", "腺鳞癌", "其他病理类型的非小细胞肺癌", "侵袭前病变", "弥漫性特发性肺神经内分泌细胞增生", "多形性癌", "梭形细胞癌", "癌肉瘤", "肺母细胞瘤", "淋巴上皮瘤样癌", "NUT癌", "胸膜间皮瘤", "胸腺癌", "肉瘤样癌", "混合型非小细胞癌", "神经内分泌癌"],只有当这些词汇出现时才提取相关信息。
        }]
        输入报告：""",
    "基本信息": """你的任务是按要求提取报告中和‘出生日期/年龄/性别’相关的信息,直接按照输出格式输出结果，不要推理。
    输出格式: [{
        "出生日期": '' 
        "年龄": ''
        "性别": "可选项为'男','女','未知','NA'"
    }]
    输入报告：""",
    "免疫检测": """你是一个医学专家，从输入报告中提取'免疫检测'的相关信息,没有的信息输出NA。输出格式:[{
        "免疫检测日期": ''。
        "TPS":   又叫'TC'。
        "PDL1": ""
        "CPS": ""
        "IC": ""
    }//如果有多条免疫检测，继续以类似Json列出]
    输入报告：""",
    "诊断": """提取报告中'诊断'的相关信息,输出格式:[{
        "诊断医生": "多个诊断医生以列表列出"
    }]
    输入报告：
    """,
    "肿瘤治疗": """你的任务是从输入报告中提取‘肿瘤治疗’相关信息，不要根据药品推理肿瘤具体治疗方式，报告中未提及的信息输出NA。若有多条肿瘤治疗，按照输出格式依次输出，输出格式::
        [{'治疗开始日期': 若没有明确指出‘肿瘤治疗’开始日期，输出'NA'。
        '治疗结束日期': 若没有明确指出‘肿瘤治疗’的结束日期，输出'NA'
        '肿瘤具体治疗方式': 可选项为['手术','消融','胸腔灌注','心包灌注','粒子植入','介入治疗','放疗','同步放化疗''化疗','靶向','免疫','PDL1','PD1','CLT-4','溶瘤病毒','肿瘤疫苗','抗血管','内分泌','细胞疗法','器官移植','干细胞移植','CAR-T','TIL']。只有明确提到'同步放化疗'的才选'同步放化疗'
        '治疗用药名称': '以列表形式列出所有具体的药品名称'
        '手术部位': 可选项为['切肺','脑转移','肝转移','其他']
        }]
    输入报告：
    """,
    "影像学": """提取输入报告中'影像学检查'的脑转部位相关信息，报告中明确出现“转移”二字才做对应选项的输出，若提取不到则不输出。注意：若原文包含"'顶叶/额叶/颞叶/枕叶"转移字眼，则输出"大脑半球转移"。报告中所有“疑似/考虑转移”不等同于转移，请不要输出。
    输出格式:[{
        "脑转移日期": '' 
        "脑转部位": ' 可选项为['大脑半球转移','丘脑转移','小脑转移','脑干转移','脑膜转移','未知','脑转移','无脑转']，该选项可多选，若有多个以列表形式输出。只有报告中明确出现“无脑转”或"未明显出现转 移"，"未知"等字眼，才输出相对应选项。
    }]//若有多条不同日期的影像学记录，继续以相同格式列出，并放到列表中.
    输入报告：""",
    "疾病": """"你的任务是提取报告中和‘疾病’相关的信息,按照要求输出结果。输出格式:[{
        "疾病首次确诊日期": ' '
        "第一次病理确诊时间（穿刺、术后病理等）": ' '
        "第一次切肺手术时间": ' '
        "第一次影像确诊时间": ' '
        "第一次治疗时间（药物、放疗等）": '   '     
        "首发症状时间": ' '
        "疾病名称": ' 可选项为'非小细胞肺癌','小细胞肺癌','胸膜间皮瘤','胸腺癌','腺癌','鳞屑样腺癌','腺泡样腺癌','乳头状腺癌','微乳头状腺癌','实性腺癌','浸润性黏液腺癌','混合性浸润性黏液性和非黏液性腺癌','胶体样腺癌','胎儿型腺癌','肠型腺癌','微浸润性腺癌','原位腺癌','黏液表皮样癌','腺样囊性癌','唾液腺型肿瘤','上皮肌上皮癌','鳞癌','角化型鳞状细胞癌','非角化型鳞状细胞癌','基底样鳞状细胞癌','原位鳞状细胞癌','大细胞癌','巨细胞癌','小细胞癌','复合型小细胞癌','腺鳞癌','侵 袭前病变','弥漫性特发性肺神经内分泌细胞增生','肉瘤样癌','多形性癌','梭形细胞癌','癌肉瘤','肺母细胞瘤','淋巴上皮瘤样癌','NUT癌','其他病理类型的非小细胞肺癌', '胸膜间皮瘤', '胸腺癌', '混合型非小细胞肺癌', '神经内分泌癌', '无'。'
    }]""",
    "治疗用药方案": """你的任务是从输入报告中提取‘治疗用药方案’相关信息，报告中未提及的信息输出NA，出现连续的日期，首个日期为开始日期，最后一个日期为结束日期。若有多条治疗用药方案，按治疗日期分组继续以相同方式列出，没有治疗日期按照每次用药输出，不要遗漏。，按照输出格式依次输出，输出格式::
    [{
    治疗开始日期: '' //用药的开始日期，不要推断开始日期，若直接说明当日用药，提取该日期，否则输出NA。
    治疗结束日期: ''//用药的结束日期，不要推断结束日期，明确说明什么时候结束再提取，否则输出NA。
    治疗用药名称: '' // 以列表形式列出，已停止使用的药不输出；原文简写的治疗方案直接输出，如‘**治疗方案’、‘**方案’直接输出‘**’等。
    治疗用药是否为建议: '' //治疗用药名称是否为建议用药,明确说明建议使用某些药输出‘是’，否则输出‘否’，没有用药信息输出NA。
    }]
    输入报告：
    """,
    "基因检测": """你的任务是提取报告中和‘基因检测’相关的信息,按照输出格式的要求输出结果。特别注意：如果报告中的基因检测结果明确提到了突变+,但没有具体说明突变的类型，请选择“阳性”，如果提到了突变的类型，请直接选择突变的类型。
    :[{
        "基因检测日期": ' 
        "EGFR": ' 选项为["扩增", "18阳性", "19阳性", "20阳性", "21阳性", "19del", "19del/L858R", "L858R", "T790M", "20插入", "C797S", "G719X", "G719A", "G719D", "G719S", "G719C", "18del", "19ins", "S768I", "L861Q", "E709V", "其他罕见突变", "阳性"],19外显子突变”应等同于19del。
        "ALK": ' 选项为["融合", "点突变", "重排", "插入", "扩增", "易位", "阳性"]
        "KRAS": ' 选项为["扩增", "G13C", "G12A", "G12C", "G12X", "G213X", "Q61X", "G12D", "G12V", "G13D", "Q61L", "其他罕见突变", "阳性"]
        "BRAF": ' 选项为['V600E','非V600','阳性']
        "MET": ' 选项为['MET14跳突','MET扩增','c-MET过表达','c-MET扩增','其他罕见突变','MET14插入','MET14缺失','MET融合','阳性'],请注意'c-MET扩增'与'MET扩增'的区别,严格选项，切勿混淆。      
        "RET": ' 选项为['融合','点突变','重排','插入','缺失','阳性'],“RET 17外显子突变”需要选择“点突变”。
        "ROS1": ' 选项为['融合','点突变','重排','G3032R','S1986F','S1986Y','阳性']
        "NTRK": '["阳性", "融合", "点突变", "重排", "NTRK1", "NTRK2", "NTRK3"] “NTRK”是一个分层选项，例如出现“NTRK1 阳点突变/融合”，输出结果为“NTRK1 、点突变、融合”。
        "HER2(ERBB2)": ' 选项为['20插入','非20插入','阳性']
        "FGFR": ' 选项为['FGFR1','FGFR2','FGFR3','FGFR4','点突变','融合','重排','易位','阳性']，如病历中出现“FGFR1”，输出结果应该为'FGFR1 。
        "BRCA": ' 选项为['BRCA1','BRCA2','阳性']
        "TP53": ' 选项为['阳性']
        "KEAP1": ' 选项为['阳性']
        "STK11": ' 选项为['阳性']
        "HER4（ERBB4）": ' 选项为['阳性']
        "RB1": ' 选项为['阳性']
        "HER3（ERBB3）": ' 选项为['阳性']
        "患者是否进行基因检测": ' 选项为['是','否']
    }]""",
    "合并疾病": """"你的任务是提取该患者的合并疾病相关信息。如果有该患者患多个合并疾病，需要你将每个疾病结果创建为一个JSON对象，将多个JSON对象放到一个列表中。例如“高血压”在'既往史','诊断'中均出现，你需要针对高血压这个疾病名称输出两个JSON对象，信息来源分别为既往史','诊断'。合并疾病":[{
        "合并疾病确诊日期": ''
        "信息来源": ' 疾病的信息来源于报告哪个部分，选项为['现病史','既往史','诊断','出院诊断','入院诊断']
        "合并疾病": ' 选项为:["重症肌无力", "门静脉左侧分支癌栓", "虹膜炎", "上肢深静脉血栓", "梅毒螺旋体感染", "狼疮肾炎", "内脏器官浅静脉血栓", "动脉硬化闭塞症", "冠心病", "甲状旁腺功能减退症", "川崎病", "结直肠癌", "痛风", "内脏器官深静脉血栓", "特应/异皮炎（湿疹）", "肺不张", "阿尔茨海默病", "银屑病", "膈肌以上下腔静脉内", "卵巢癌", "甲状旁腺功能亢进症", "高血压", "癌栓累及肠系膜上腔静脉", "干燥综合征", "胃癌", "颅内静脉血栓", "急性冠状动脉综合征", "溃疡改变", "肾癌", "血胸", "肾囊肿", "霍奇金淋巴瘤", "白癜风", "肿瘤破裂 （血管包绕）", "格林-巴利综合征", "溃疡性结肠炎", "系统性红斑狼疮", "肺气肿", "白内障", "甲型肝炎", "肾积水", "糖尿病", "胆管癌栓", "乳腺癌", "非霍奇金淋巴瘤", "宫颈癌", "卒中", "心力衰竭", "周围神经炎", "消化道造瘘", "结膜炎", "肝静脉主干癌栓", "严重肾功能不全", "食管胃底静脉曲张", "高血脂", "甲状腺癌", "门静脉主干癌栓", "鼻咽癌", "穿孔", "脂肪肝", "腹水", "硬脑膜动脉瘘", "血栓性血小板减少性紫癜", "下肢浅静脉血栓", "心脏受累", "泌尿生殖系统出血", "心功能不全", "肺大疱", "痴呆", "干眼症", "视网膜脱落", "阻塞性肺气肿", "胰腺癌", "葡萄膜炎", "食管癌", "美尼尔病", "动脉粥样硬化", "脓胸", "气胸", "肺纤维化", "癫痫", "炎症", "弥散性血管内凝血", "癌栓累及肠系膜下腔静脉", "上腔静脉压塞综合征", "下腔静脉内", "肺炎", "类风湿关节炎", "I型糖尿病", "精神疾病", "管腔狭窄", "门静脉右侧分支癌栓", "肝硬化", "子宫内膜癌", "精神分裂", "肾性贫血", "内脏器官动脉血栓", "泌尿生殖系统炎症", "高血压肾病", "严重心律失常", "眼结石", "上肢浅静脉血栓", "间质性肺炎", "溃烂", "乙型肝炎", "肺源性心脏病", "溃疡", "子痫", "消化道结石", "尿毒症", "肝性脑病", "肺水肿", "动脉斑块", "结核", "黄斑水肿", "软化", "肝静脉分支癌栓", "人乳头状瘤病毒感染", "脊髓压迫", "艾滋病", "双向情感障碍", "肝癌", "上肢动脉血栓", "膀胱癌 ", "癌性血栓性静脉炎", "荨麻疹", "自身免疫性疾病", "血栓", "多发性硬化症", "高尿酸血症", "肺感染", "出血", "肢体/脏器缺血性改变", "白血病", "丙型肝炎", "颅内动脉血栓", "脑梗", "克罗恩病", "腹泻", "下肢动脉血栓", "下肢深静脉血栓", "梗阻", "青光眼", "感染", "痒疹", "抑郁症", "黄斑变性", "NA", "慢性支气管炎", "帕金森", "栓塞（肺栓塞）", "食管裂孔疝", "鼻饲", "肾功能衰竭", "埂塞"]
    }//如果有多条，继续以相同方式列出]。
    输入报告""",
    "日期": """从输入报告中提取与日期相关的信息,按照以下要求输出结果,未提及的日期输出NA，请不要推理。输出格式:{
        "入院日期": '',
        "出院日期": '',
        "病史采集日期": '',
        "记录日期": '',
        "就诊日期": '',
        "检查日期": '',
        "报告日期": '',
        "审核日期": '',
        "送检日期": '',
        "收到日期": '',
        "申请日期": '',
        "会诊日期": ''
        }
        输入报告：""",
}
