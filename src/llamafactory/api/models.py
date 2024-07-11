from typing import List

from ..data import Role as DataRole
from ..extras.logging import get_logger
from .common import dictify, jsonify
from .protocol import Role, ModelList, SingleReportResponse
import pandas
from ..chat import ChatModel
from parse_ds.sft_prompt import sft_unit_prompt

import json

logger = get_logger(__name__)
ROLE_MAPPING = {
    Role.USER: DataRole.USER.value,
    Role.ASSISTANT: DataRole.ASSISTANT.value,
    Role.SYSTEM: DataRole.SYSTEM.value,
    Role.FUNCTION: DataRole.FUNCTION.value,
    Role.TOOL: DataRole.OBSERVATION.value,
}
# 调用名：配置
model_conf_path = "src/llamafactory/api/api_config/model_conf.json"

MODEL_MAPPING = pandas.read_json(model_conf_path, orient="records").to_dict(orient="records")


def _model_list() -> List[dict]:
    model_list = MODEL_MAPPING
    return model_list
def single_report_extract(unit_name, report,model:"ChatModel"):   
    """单篇报告提取"""
    prompt = sft_unit_prompt.get(unit_name)
    if not prompt:
        return SingleReportResponse(unit_json={"msg":"单元名不存在"})
    messages = []
    query = f"{prompt}\n{report}"
    messages.append({"role": "user", "content": query})
    response = ""
    response = model.chat(messages)
    response = response[0].response_text
    try:
        unit_json = json.loads(response)
    except Exception as e:
        unit_json = {
            "response": response,
            "error": str(e),
        }
    return SingleReportResponse(unit_json=unit_json)
