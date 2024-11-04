# overall_version： 来源于客户需求：如 ss肺癌-V2 ss肺癌-V3 —— 不单独管理，程序员添加
# disease：疾病库，树形结构，每个客户结构不一样
# promptwork：prompt模板库，多种疾病共用一套模板
# report：报告类型
# unit：单元，包括prompt和schema
# version: 单元的迭代版本号，重要的是inuse

import json
import os
import sys
import time
import traceback
from datetime import datetime

from peewee import *
from playhouse.shortcuts import ReconnectMixin

from utils.conf import db_conf


class ReconnectMySQLDatabase(ReconnectMixin, MySQLDatabase):
    pass


db_conn = ReconnectMySQLDatabase(
    # db_conn = MySQLDatabase(
    db_conf["database"],
    host=db_conf["host"],
    port=db_conf["port"],
    user=db_conf["username"],
    passwd=db_conf["password"],
    # charset=db_conf["db_charset"],
)


class JSONField(TextField):
    def db_value(self, value):
        return json.dumps(value) if value else None

    def python_value(self, value):
        if value is not None:
            return json.loads(value)


class BaseModel(Model):
    class Meta:
        database = db_conn


class ds_image(BaseModel):
    id = AutoField()
    patient_id = BigIntegerField()
    url = CharField(max_length=255)
    url_origin = CharField(max_length=255)
    url_desensitive = CharField(max_length=255)
    scale_factor = DoubleField()
    ocr_result = TextField()
    ocr_raw = TextField()
    category = CharField(max_length=64)
    preprocess = TextField()
    native_result = JSONField()
    native_result_custom = JSONField()
    pic_result = JSONField()
    pic_result_custom = JSONField()
    description = TextField()
    status = IntegerField()
    creator_id = BigIntegerField()
    modifier = CharField(max_length=255)
    dept_belong_id = BigIntegerField()
    update_datetime = DateTimeField()
    create_datetime = DateTimeField()


class ds_dataset(BaseModel):
    id = AutoField()
    name = CharField()
    stage = IntegerField()
    status = IntegerField()
    description = TextField()

class ds_patient(BaseModel):
    id = AutoField()
    dataset_id = IntegerField()
    status = IntegerField()
