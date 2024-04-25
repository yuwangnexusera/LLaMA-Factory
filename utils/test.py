import json
import sys
from  datetime import datetime
sys.path.append(".")
from utils.google_translate import translate_text
def mapping_loc_zh_en(key,trans=True):
    with open("utils/mapping_answer_zh_en.json", "r", encoding="utf-8") as f:
        mapping = json.load(f)
    return_key = mapping.get(key)
    if return_key:
        return return_key
    elif trans:
        print(f"{datetime.now()}-translate{key}")
        return translate_text(key)
    else:
        return key
def process_single_loc_obj(dic):
    if isinstance(dic, dict):
        new_dic = {}
        for k, v in dic.items():
            new_key = mapping_loc_zh_en(k)  # 如果key不存在于maps中，保留原始键
            if new_key:
                if isinstance(v, list):
                    # 只映射列表中存在于maps的值
                    new_dic[new_key] = [mapping_loc_zh_en(v_i) for v_i in v]  # 默认为原始值
                elif "日期" in k or "时间" in k or k=="NA":
                    new_dic[new_key] = v
                else:
                    # 映射值，如果值不存在于maps中，使用原始值
                    new_dic[new_key] = mapping_loc_zh_en(v)
        return new_dic  # 返回新字典代替修改原字典
    elif isinstance(dic, list):
        return [process_single_loc_obj(d) for d in dic]  # 使用列表推导处理列表中的每个元素
    else:
        raise ValueError("not support type")


if __name__ == '__main__':

    with open('nex_dataset/test/extract_with_unit.json', 'r', encoding='utf-8') as f:
        # TODO 处理可多条的情况，训练数据需要补充
        data = json.load(f)
        for item_test in data:
            new_item = {}
            output = json.loads(item_test["output"])
            for unit, locs in output.items():
                en_unit = mapping_loc_zh_en(unit)
                if en_unit and en_unit not in new_item:
                    new_item[en_unit] = {}
                processed_locs = process_single_loc_obj(locs)
                new_item[en_unit] = processed_locs
            item_test["output"] = json.dumps(new_item)
    with open("nex_dataset/test/extract_with_unit.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
