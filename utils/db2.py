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


class hx_ocr_result(BaseModel):
    url = CharField()
    ocr_engine = CharField()
    ocr_raw = JSONField()
    ocr_result = JSONField()
    create_time = DateTimeField(default=datetime.now)


class hx_pic_result(BaseModel):
    user_id = CharField()
    pic_id = CharField()
    image_url = CharField()
    image_url_desensitive = CharField()
    scale_factor = FloatField(default=1)

    ocr_res = JSONField(null=True)
    pic_res = TextField()
    pic_res_custom = JSONField()
    cache_key = CharField()

    hash_code = CharField()
    token_in = IntegerField(default=0)
    token_out = IntegerField(default=0)
    total_token = IntegerField(default=0)
    token_cost = FloatField(default=0)
    token_in_cost = FloatField(default=0)
    token_out_cost = FloatField(default=0)
    channel = CharField()
    create_time = DateTimeField(default=datetime.now)


class hx_crf_result(BaseModel):
    creation_time = DateTimeField(default=datetime.now)
    crf_json_string = TextField()
    custom_crf = JSONField()
    update_time = DateTimeField()
    user_id = CharField()
    series = CharField()
    summary = TextField()


class hx_crf_err(BaseModel):
    creation_time = DateTimeField(default=datetime.now)
    ex = TextField()
    pic_id = CharField()
    prompt = TextField()
    user_id = CharField()
    series = CharField()


class crf_request(BaseModel):
    channel = CharField()
    creation_time = DateTimeField(default=datetime.now)
    ip = TextField()
    user_id = CharField()
    series = CharField()
    request = TextField()
    request_success_count_by_pic_id = IntegerField(default=0)
    request_total_count_by_pic_id = IntegerField(default=0)
    disease_type = CharField()
    result_success_count_by_user_id = IntegerField(default=0)
    result_total_count_by_user_id = IntegerField(default=0)
    update_time = DateTimeField(default=datetime.now)


class hx_analysis(BaseModel):
    accomplished = IntegerField(default=0)
    channel = CharField()
    creation_time = DateTimeField(default=datetime.now)
    pic_id = CharField()
    user_id = CharField()


class ss_prompt(BaseModel):
    id = AutoField()
    report_type = IntegerField()
    promptwork_name = CharField()
    model_name = CharField()
    unit_name = CharField()
    schemas = JSONField()
    template = TextField()
    creation_time = DateTimeField(default=datetime.now)
    version = CharField()
    status = IntegerField(default=1)
    overall_version = CharField()
    test_version = CharField()


class ss_prompt_step(BaseModel):
    id = AutoField()
    report_type = IntegerField()
    model_name = CharField()
    unit_name = CharField()
    location = CharField()
    step = CharField()
    promptwork_name = CharField()
    schemas = JSONField()
    template = TextField()
    version = CharField()
    creation_time = DateTimeField(default=datetime.now)
    overall_version = CharField()
    status = IntegerField(default=1)


# 所有报告类型
class ss_report(BaseModel):
    id = AutoField()
    report_type = CharField()
    description = CharField()
    removed = IntegerField(default=0)
    list_order = IntegerField(default=9999)
    update_time = DateTimeField(default=datetime.now)
    create_time = DateTimeField(default=datetime.now)


# 报告类型下，所有可能的单元
class ss_report_unit(BaseModel):
    id = AutoField()
    report_type = CharField()
    unit_name = CharField()
    removed = IntegerField(default=0)
    update_time = DateTimeField(default=datetime.now)
    create_time = DateTimeField(default=datetime.now)


# 模板下关联的报告
class ss_report_type(BaseModel):
    id = AutoField()
    report_type = CharField()
    description = CharField()
    status = IntegerField(default=1)
    version = CharField()
    units = CharField()
    update_time = DateTimeField(default=datetime.now)

    promptwork_name = CharField()


class ss_temp_prompt(BaseModel):
    id = AutoField()
    report_type = IntegerField()
    model_name = CharField()
    unit_name = CharField()
    schemas = JSONField()
    template = TextField()
    recall_rate = CharField()
    precise_rate = CharField()
    version = CharField()
    create_time = DateTimeField(default=datetime.now)
    correct_answer = JSONField()
    gpt_answer = JSONField()
    error_details = JSONField()
    status = IntegerField(default=0)
    overall_version = CharField()

    promptwork_name = CharField()


class ss_slot_schema(BaseModel):
    name = CharField()
    level = CharField()
    category = CharField()
    data_type = CharField()
    domain = JSONField()
    update_time = DateTimeField()
    create_time = DateTimeField(default=datetime.now)


class ss_slot_alias(BaseModel):
    name = CharField()
    alias = CharField()
    create_time = DateTimeField(default=datetime.now)


class ss_turmo_medicine(BaseModel):
    id = AutoField()
    approval_number = CharField()
    medicine_category = CharField()
    component_name = CharField()
    component_synonym = CharField()
    general_name = CharField()
    product_name = CharField()
    product_synonym = CharField()
    production_address = CharField()
    specification = CharField()
    is_market = CharField()


class ss_patient_test(BaseModel):
    id = AutoField()
    patient_id = CharField()
    pic_native_json = JSONField()
    pic_post_json = JSONField()
    patient_res = JSONField()
    precise_rate = CharField()
    recall_rate = CharField()
    differences = JSONField()
    create_time = DateTimeField(default=datetime.now)


class ss_promptwork(BaseModel):
    id = AutoField()
    name = CharField()
    note = CharField()
    inuse_model = JSONField()
    update_time = DateTimeField(default=datetime.now)
    create_time = DateTimeField(default=datetime.now)


class ss_disease(BaseModel):
    id = AutoField()
    overall_version = CharField()
    pid = IntegerField()
    is_diagnose = IntegerField(default=0)
    name = CharField()
    promptwork_name = CharField()
    update_time = DateTimeField(default=datetime.now)
    create_time = DateTimeField(default=datetime.now)


class ss_prompt_overall_version(BaseModel):
    id = AutoField()
    name = CharField()
    update_time = DateTimeField(default=datetime.now)
    create_time = DateTimeField(default=datetime.now)


class ss_unit_dataset(BaseModel):
    id = AutoField()
    report_id = IntegerField(null=True)
    url = CharField(max_length=255, null=True)
    content = TextField(null=True)
    unit_name = CharField(max_length=255, null=True)
    pic_res = JSONField(null=True)
    res_custom = JSONField(null=True)
    set_type = IntegerField()
    update_time = DateTimeField(default=datetime.now)
    model_name = CharField(max_length=255, null=True)
    overall_version = CharField(max_length=255, null=True)
    disease = CharField(max_length=100, null=True)
