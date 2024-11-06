full_dose_prompt = {
    "基因检测": """你是一名出色的基因检测专家，具备出色的专业技能，能够准确解读医学报告中的各种医疗数据。通过临床经验和深厚的医学知识，你能够精准地识别并理解报告中的基因检测信息。注意报告中是以日期为分隔的段落。
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
 "患者是否进行基因检测": ['是','否']
 "基因检测": "当你判断出他做了基因检测时,就直接提取基因的名称和突变类型。注意:所有提取的信息必须直接来源于报告文本,不允许基于文本内容进行任何形式的推断、假设或改变。"
}]
""",
    "合并疾病": """我在进行药品临床试验的患者筛选工作，需要提取肿瘤患者有关“伴随疾病”的信息。从下述报告中提取到有关患者的内分泌及免疫系统、神经系统疾病、消化系统疾病、呼吸系统疾病、循环系统疾病、传染性疾病、运动系统疾病、恶性肿瘤情况、泌尿生殖系统疾病、眼耳鼻喉相关疾病等系统的伴随疾病。若有多条不同日期的伴随疾病，需要你将每个伴随疾病创建为一个JSON对象，如果有多个伴随疾病，就将多个JSON对象合并到一个列表中，切勿遗漏。

注意1：如果报告中某一疾病名称出现多次，你需要区分不同信息来源再输出，例如“高血压”在'既往史','诊断'中均出现，你需要针对高血压这个疾病名称输出两个JSON对象，信息来源分别为既往史','诊断'。
注意2：所有提取的信息必须直接来源于报告文本，不允许基于文本内容进行任何形式的推断、假设或改变。
注意3：请确认所提取疾病确实为疾病
注意4：伴随疾病我们想要除了 “肺癌” 外的疾病
注意5：请确认不要遗漏疾病，输出完整
注意6：基因不是疾病的一种。
输入报告: |||input_report|||

输出格式: The output should be a markdown code snippet formatted in the following schema, including the leading and trailing ```json and ```:\
[{
 "伴随疾病确诊日期": string  //输出格式为'%Y-%m-%d'。如果前一个选项“信息来源”输出为“诊断”，则该选项输出患者的就诊时间；如果为“出院诊断”，则该选项输出患者的出院时间；如果为“入院诊断”，则该选项输出患者的入院时间。
 "信息来源": string  // 疾病的信息来源于报告哪个部分，在报告中明确出现以下可选项时输出对应选项。选项为['现病史','既往史','诊断','出院诊断','入院诊断']
 "伴随疾病名称": string  // 
}]""",
    "病理": """你的任务是从下面的‘医学报告’中提取所有【肺癌】病理检查分型\
        临床诊断中的病理类型也应被提取。需要你将每次病理检测的结果创建为一个JSON对象。JSON对象中的键和值的详细描述在下面的‘输出格式’中；\
            如果报告中有多次病理检测，就将多个JSON对象放到一个列表中。请注意，只关注报告中明确提及的病理检测结果，而不是基于影像学检查（如CT扫描、PET/CT等）的诊断。\
                请忽略任何基因检测、影像学检查和其他非病理学的医学检查结果。
注意报告中是以日期为分隔的段落。与日期相关的字段格式为'%Y-%m-%d'，如果年/月/日中的任何一项或多项缺失，用NA代替，请按照这种格式输出（例如'2023-05-NA'）。若这些日期都不可用，请输出‘NA’。
请确保信息直接从医学报告中原样提取，避免任何推理。如果某个字段在报告中未提及，请输出‘NA’。
医学报告: |||input_report|||
输出格式: The output should be a markdown code snippet formatted in the following schema, including the leading and trailing ```json and ```:
[{
 "病理日期": //输出格式为'%Y-%m-%d'。
 "病理类型": // 多个病理类型以数组形式输出，只提取医学术语中病理类型。
}]""",
    "疾病": """你的任务是按要求提取报告中所有【肺癌】疾病信息,按照要求输出结果。解析出来的结果必须直接来源于原报告,不要推理。不要输出任何说明性文字。提取不到的字段输出'NA'。
    输入报告:|||input_report|||
  输出格式: 输出应该是按照以下模式格式化的markdown代码片段，包括开头和结尾 ```json and ```
    {
 "疾病名称": //多个疾病以列表形式列出，只提取医学术语中的疾病名称
}""",
    "免疫检测": """你的任务是按要求解析一个报告中'免疫检测'记录的相关信息。注意报告中是以日期为分隔的段落，所以免疫检测日期应该输出最接近免疫检测结果描述的日期。
需要你将每次免疫组化检测的结果创建为一个JSON对象，JSON对象中的键和值的详细描述在下面的'输出格式中'；如果有多次免疫组化检测，就将多个JSON对象放到一个列表中。解析出来的结果必须直接来源于原报告,请不要通过推理、计算等得出结论。请务必不要误解成写代码,也不要输出任何说明性文字,直接输出结果。注意:提取不到的字段输出'NA'。
以下是三件非常重要的事情：
1. TPS和PDL1分别是不同的指标，请勿混为一谈，这非常重要。
2. IC和PDL1分别是不同的指标，请勿混为一谈，这非常重要。
3. 病历中没有提到TPS这个指标的时候，请勿用PDL1代替。

输入报告:|||input_report|||
输出格式: The output should be a markdown code snippet formatted in the following schema, including the leading and trailing ```json and ```
{
 "免疫检测日期": //输出格式为'%Y-%m-%d'。
 "TPS": //当有PDL1评分同时出现的时候,注意区分两者的区别,当病历中没有明确提到TPS的时候,TPS的值应该是NA,只输出TPS的数值和%、<、>等符号,不要字母。
 "CPS": //没有明确提到CPS的时候,CPS的值应该是NA,只输出CPS数值和%、<、>等符号,不要字母。
 "IPS": //没有明确提到IPS的时候,IPS的值应该是NA,只输出IPS数值和%、<、>等符号,不要字母。
  "TC": //没有明确提到TC的时候,TC的值应该是NA,只输出TC数值和%、<、>等符号,不要字母。
  "IC": //没有明确提到IC的时候,IC的值应该是NA,只输出IC数值和%、<、>等符号,不要字母。
 "PDL1": //当有TPS评分同时出现的时候,注意区分两者的区别,当病历中没有明确提到PDL1的时候,PDL1的值应该是NA,只输出PDL1数值和%、<、>等符号,不要字母。
}""",
    "治疗用药方案": """你的任务是按要求提取报告中和‘治疗用药’相关的信息,按照要求输出结果。提取不到的字段输出'NA'。注意报告中是以日期为分隔的段落。与日期相关的字段格式为'%Y-%m-%d'，如果年/月/日中的任何一项或多项缺失，报告中已用“NA”代替，直接输出即可。若有多条不同日期的治疗记录，需要你将每个治疗周期创建为一个JSON对象，将多个JSON对象放到一个列表中，切勿遗漏。
输入报告:|||input_report|||
输出格式: The output should be a markdown code snippet formatted in the following schema, including the leading and trailing ```json and ```:
[{
 "治疗开始日期": //输出格式为'%Y-%m-%d',若没有明确指出开始日期，请直接输出'NA'。
 "治疗结束日期": //输出格式为'%Y-%m-%d',若没有明确指出结束日期，请直接输出“NA”，切忌用其他日期替代。
 "治疗用药名称": //以列表形式列出，提取报告中患者治疗用药的药品名称。若提取不到用药名称，则直接输出'NA'。
 "治疗用药是否为建议": //可选项为'是','否'。不可多选。
}]""",
    "日期": """从输入报告中提取与日期相关的信息,按照以下要求输出结果,未提及的日期输出NA，请不要推理。输出格式为'%Y-%m-%d'
    输入报告：|||input_report|||
输出格式: The output should be a markdown code snippet formatted in the following schema, including the leading and trailing ```json and ```:
{"入院日期": '',"出院日期": '',"病史采集日期": '',"记录日期": '',"就诊日期": '',"检查日期": '',"报告日期": '',"审核日期": '',"送检日期": '',"收到日期": '',"申请日期": '',"会诊日期": ''}
""",
}

sft_prompt = {
    "基因检测": """你是一名出色的基因检测专家，能够精准地识别并理解报告中的基因检测信息。\
    请根据下面提供的报告解析出每次基因检测中的有关基因突变的信息。基因检测中可能包含EGFR, ALK, KRAS等基因的突变状态,请不要漏掉任何基因检测结果。
  输出格式: [{
 "基因检测日期": string  //输出格式为'%Y-%m-%d',不要推理。
 "患者是否进行基因检测": ['是','否']，如果报告中提到“基因”，输出是
 "基因检测": "当你判断出他做了基因检测时,就直接提取基因的名称和突变类型。注意:所有提取的信息必须直接来源于报告文本,不允许基于文本内容进行任何形式的推断、假设或改变。"
}]
  输入报告：
""",
    "合并疾病": """从下述报告中提取到有关患者的内分泌及免疫系统、神经系统疾病、消化系统疾病、呼吸系统疾病、循环系统疾病、传染性疾病、运动系统疾病、恶性肿瘤情况、泌尿生殖系统疾病、眼耳鼻喉相关疾病等系统的伴随疾病。\
如果报告中某一疾病名称出现多次，你需要区分不同信息来源再输出，例如“高血压”在'既往史','诊断'中均出现，你需要针对高血压这个疾病名称输出两个JSON对象，信息来源分别为既往史','诊断'。
输出格式: \
[{
 "伴随疾病确诊日期": string  //输出格式为'%Y-%m-%d'。如果前一个选项“信息来源”输出为“诊断”，则该选项输出患者的就诊时间；如果为“出院诊断”，则该选项输出患者的出院时间；如果为“入院诊断”，则该选项输出患者的入院时间。
 "信息来源": string  // 疾病的信息来源于报告哪个部分，在报告中明确出现以下可选项时输出对应选项。选项为['现病史','既往史','诊断','出院诊断','入院诊断']
 "伴随疾病名称": string  // 
}]
输入报告: """,
    "疾病": """你的任务是按要求提取报告中所有【肺癌】疾病信息,按照要求输出结果。解析出来的结果必须直接来源于原报告,不要推理。不要输出任何说明性文字。提取不到的字段输出'NA'。
  输出格式: 
    {
 "疾病名称": //多个疾病以列表形式列出，只提取医学术语中的疾病名称
}
 输入报告：""",
    "病理": """你的任务是从下面的‘医学报告’中提取所有【肺癌】病理检查分型,临床诊断中的病理类型也应被提取。\
           请注意，只关注报告中明确提及的病理检测结果，而不是基于影像学检查（如CT扫描、PET/CT等）的诊断。请忽略任何基因检测、影像学检查和其他非病理学的医学检查结果。
请确保信息直接从医学报告中原样提取，避免任何推理。如果某个字段在报告中未提及，请输出‘NA’
输出格式:
[{
 "病理日期": //输出格式为'%Y-%m-%d'。
 "病理类型": // 多个病理类型以数组形式输出，只提取医学术语中病理类型。
}]
医学报告: """,
    "治疗用药方案": """你的任务是从输入报告中提取‘治疗用药方案’相关信息，提取报告中所有的用药记录,不要遗漏。报告中未提及的信息输出NA，停用的药不提取。输出格式:\n    [{\n    治疗开始日期: '' \/\/用药的开始日期，直接说明再提取，不要推断，否则输出NA。\n    治疗结束日期: ''\/\/用药的结束日期，明确说明什么时候结束用药、完成用药再提取不要推断，否则输出NA。\n    治疗用药名称: '' \/\/ 以列表形式列出单个日期或者单次治疗的所有药品。原文中如‘XXX治疗方案’、‘XXX方案’等描述直接输出‘XXX’,不要从XXX推理出药品。\n    治疗用药是否为建议: '' \/\/明确说明建议使用某些药输出‘是’，否则输出‘否’，没有用药信息时输出NA。\n    }]\n    输入报告：\n    """,
    "日期": """从输入报告中提取与日期相关的信息,按照以下要求输出结果,未提及的日期输出NA，请不要推理。输出格式:{\n\"入院日期\": '',\n\"出院日期\": '',\n\"病史采集日期\": '',\n\"记录日期\": '',\n\"就诊日期\": '',\n\"检查日期\": '',\n\"报告日期\": '',\n\"审核日期\": '',\n\"送检日期\": '',\n\"收到日期\": '',\n\"申请日期\": '',\n\"会诊日期\": ''\n}\n输入报告：""",
    "免疫检测": """你的任务是按要求解析一个报告中'免疫检测'记录的相关信息。解析出来的结果必须直接来源于原报告,请不要通过推理、计算等得出结论。请务必不要误解成写代码,也不要输出任何说明性文字,直接输出结果。注意:提取不到的字段输出'NA'。
输出格式: 
{
 "免疫检测日期": //输出格式为'%Y-%m-%d'。
 "TPS": //当有PDL1评分同时出现的时候,注意区分两者的区别,当病历中没有明确提到TPS的时候,TPS的值应该是NA,只输出TPS的数值和%、<、>等符号,不要字母。
 "CPS": //没有明确提到CPS的时候,CPS的值应该是NA,只输出CPS数值和%、<、>等符号,不要字母。
 "IPS": //没有明确提到IPS的时候,IPS的值应该是NA,只输出IPS数值和%、<、>等符号,不要字母。
  "TC": //没有明确提到TC的时候,TC的值应该是NA,只输出TC数值和%、<、>等符号,不要字母。
  "IC": //没有明确提到IC的时候,IC的值应该是NA,只输出IC数值和%、<、>等符号,不要字母。
 "PDL1": //当有TPS评分同时出现的时候,注意区分两者的区别,当病历中没有明确提到PDL1的时候,PDL1的值应该是NA,只输出PDL1数值和%、<、>等符号,不要字母。
}
输入报告:
""",
    "报告分类": """你的任务是判断下列医学报告的分类。输出要求：从输出可选项中选择与‘医学报告’最匹配的一项，如果无法判定属于哪个类别输出'其他'，不要输出解释、选择的原因等说明性文字或符号。
                输出可选项及其说明：\
   {"出入院记录":,\
    "门诊病历": 病例记录和门诊记录也输出门诊病历类别,\
    "检查记录"：,\
    "基因检测",\
    "病理报告",\
    "病理会诊记录",\
    "出入院疾病诊断书":'出入院诊断证明'也输出为‘出入院疾病诊断书’,\
    "门诊疾病诊断书":'门诊诊断证明'也输出为‘门诊疾病诊断书’,\
    "其他疾病诊断书":'其他诊断证明'也输出为‘其他疾病诊断书’,\
    "检验记录":化验单也输出为‘检验记录’,\
    "其他会诊记录":除病理会诊记录外的其他会诊记录\
    "处方单",\
    "费用单",\
    "治疗记录",\
    "手术记录",\
    "医嘱单",\
    "知情同意书",\
    "体检报告",\
    "注射单",\
    "病程记录",\
    "其他"\
}
    医学报告：
        """,
}
domain_prompt = {
    "基因检测": """你是一名出色的基因检测专家，具备出色的专业技能，能够准确解读医学报告中的各种医疗数据。
请根据下面提供的'输入报告'解析出每次基因检测中的有关基因突变的信息。如果报告中没有提及某个基因的信息，请在相应字段中填写'NA'。请确保所有的信息直接来源于报告，并避免任何推理。
特别注意：如果报告中的基因检测结果明确表明突变了但没有具体说明突变的类型，判断为阳性，如果提到了突变的类型，请直接选择突变的类型，请务必只输出可选项中的答案，切勿输出可选项以外的答案。
如果一个阳性基因有多个突变类型，请选择所有正确的选项。
'TP53','KEAP1','STK11','HER4（ERBB4）','RB1'这五个基因，如果报告中明确提到是阳性，请直接选择'阳性。

输入报告：|||report|||
输出格式: The output should be a markdown code snippet formatted in the following schema, including the leading and trailing "json" and "":
  {
 "EGFR": 可选项为：['扩增','18阳性','19阳性','20阳性','21阳性','19del','19del/L858R','L858R','T790M','20插入', 'C797S', 'G719X', 'G719A', 'G719D', 'G719S', 'G719C', '18del', '19ins','S768I', 'L861Q','E709V','其他罕见突变','阳性','NA']，可多选。19外显子突变应等同于19del;
"ALK": 可选项为：['融合','点突变','重排','插入','扩增','易位','阳性','NA']，可多选
"KRAS": 可选项为：['扩增','G13C','G12A','G12C','G12X','G213X','Q61X','G12D','G12V','G13D','Q61L','其他罕见突变','阳性','NA']，可多选
"BRAF": 可选项为：['V600E','非V600','阳性','NA']，可多选
"MET": 可选项为：['MET14跳突','MET扩增','c-MET过表达','c-MET扩增','其他罕见突变','MET14插入','MET14缺失','MET融合','阳性','NA']，可多选,请注意'c-MET扩增'与'MET扩增'的区别,严格选项，切勿混淆。
"RET":选项为['融合','点突变','重排','插入','缺失','阳性','NA'],“RET 17外显子突变”需要选择“点突变”，可多选。
"ROS1": 选项为['融合','点突变','重排','G3032R','S1986F','S1986Y','阳性','NA']，可多选
"NTRK": 可选项为：['NTRK1','NTRK2','NTRK3','阳性','融合','点突变','重排','NA']，可多选
"HER2(ERBB2)": 可选项为：['20插入','非20插入','阳性','NA']，可多选
"FGFR": 可选项为：['FGFR1','FGFR2','FGFR3','FGFR4','点突变','融合','重排','易位','阳性','NA']，可多选。
"BRCA": 可选项为：['BRCA1','BRCA2','阳性','NA'] ，可多选
"TP53": 可选项为：['阳性','NA']
"KEAP1": 可选项为：['阳性','NA']
"STK11": 可选项为：['阳性','NA']
"HER4（ERBB4）": 可选项为：['阳性','NA']
"RB1":可选项为：['阳性','NA']
"HER3（ERBB3）": 可选项为：['阳性','NA']
}
请在输出前再次检查：
1. 检查输出是否正确，尤其是当没有明确提及TP53是阳性的话，请勿将TP53输出为阳性。
2. 请再检查一遍你的输出都是可选项中的选项。
3. 如果某个基因发生多种突变，请都列到列表中。
4. 请确认你没有漏掉或多提取基因检测结果。
"""
}
sft_prompt_stage2 = {
    "基因检测": """你的任务是解析下面‘输入文本’中的基因突变的信息。确保所有的信息直接来源于输入文本，避免任何推理。
特别注意：1.如果某基因检测的结果显示突变, 但没有明确说明突变的类型, 该基因点位的值就判定为阳性；\
    2.如果明确提到突变类型, 请直接从可选项中选择突变的类型, 如果基因点位的突变类型不在选项中输出'NA'。\
    3.如果一个阳性基因有多个突变类型，请选择所有正确的选项并以列表形式输出。
不在输出格式中的基因名称不提取。\
  输出格式: The output should be a markdown code snippet formatted in the following schema, including the leading and trailing ```json and ```:
  {
 "EGFR": 可选项为：['扩增','18阳性','19阳性','20阳性','21阳性','19del','19del/L858R','L858R','T790M','20插入', 'C797S', 'G719X', 'G719A', 'G719D', 'G719S', 'G719C', '18del', '19ins','S768I', 'L861Q','E709V','其他罕见突变','阳性','NA'],19外显子突变应等同于19del。
"ALK": 可选项为：['融合','点突变','重排','插入','扩增','易位','阳性','NA']
"KRAS": 可选项为：['扩增','G13C','G12A','G12C','G12X','G213X','Q61X','G12D','G12V','G13D','Q61L','其他罕见突变','阳性','NA']
"BRAF": 可选项为：['V600E','非V600','阳性','NA']
"MET": 可选项为：['MET14跳突','MET扩增','c-MET过表达','c-MET扩增','其他罕见突变','MET14插入','MET14缺失','MET融合','阳性','NA'],请注意'c-MET扩增'与'MET扩增'的区别,严格选项，切勿混淆。
"RET":选项为['融合','点突变','重排','插入','缺失','阳性','NA'],“RET 17外显子突变”需要选择“点突变”。
"ROS1": 选项为['融合','点突变','重排','G3032R','S1986F','S1986Y','阳性','NA']
"NTRK": 可选项为：['NTRK1','NTRK2','NTRK3','阳性','融合','点突变','重排','NA']
"HER2(ERBB2)": 可选项为：['20插入','非20插入','阳性','NA']
"FGFR": 可选项为：['FGFR1','FGFR2','FGFR3','FGFR4','点突变','融合','重排','易位','阳性','NA']。
"BRCA": 可选项为：['BRCA1','BRCA2','阳性','NA']
"TP53": 可选项为：['阳性','NA']
"KEAP1": 可选项为：['阳性','NA']
"STK11": 可选项为：['阳性','NA']
"HER4（ERBB4）": 可选项为：['阳性','NA']
"RB1":可选项为：['阳性','NA']
"HER3（ERBB3）": 可选项为：['阳性','NA']
}
  输入文本：
"""
}