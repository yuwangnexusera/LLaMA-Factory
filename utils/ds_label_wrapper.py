import sys
from peewee import fn, SQL
import re

sys.path.append(".")
from utils.ds_label import ds_image, ds_patient, ds_dataset


def get_obj_by_id(id):
    query = ds_image.get_by_id(id)
    return query


# 查询验证集 test开头的数据
def select_test():
    # 构建 SQL 查询 (简易版，看看patient_id的范围)
    query = ds_image.select().where((ds_image.patient_id.between(62, 76)) & (ds_image.status == 0))
    return query


# 查询训练、测试集，ds2000_开头的数据  patient_id关联【62-76】
def select_sft_train_ds():
    sft_patient_id = [2, 5, 6, 7, 8, 9, 33, 48, 53, 54, 55, 56, 57, 58, 59, 35, 36, 37, 34, 38, 45, 21, 60]
    query = ds_image.select().where((ds_image.patient_id.in_(sft_patient_id)) & (ds_image.status == 0))
    return query


if __name__ == "__main__":
    # 假设这是数据库中的一些示例数据
    test_ds = select_test()
    for dataset in test_ds:
        print(dataset.url)
    for i in test_ds:
        print(i)
