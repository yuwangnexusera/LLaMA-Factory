from typing import  List

from ..data import Role as DataRole
from ..extras.logging import get_logger
from .common import dictify, jsonify
from .protocol import (
    Role,
    ModelList,
)


logger = get_logger(__name__)
ROLE_MAPPING = {
    Role.USER: DataRole.USER.value,
    Role.ASSISTANT: DataRole.ASSISTANT.value,
    Role.SYSTEM: DataRole.SYSTEM.value,
    Role.FUNCTION: DataRole.FUNCTION.value,
    Role.TOOL: DataRole.OBSERVATION.value,
}
# 调用名：配置
MODEL_MAPPING = [
    {
        "model_alias": "qwen2-7B-chat",
        "template": "qwen",
        "model_name_or_path": "../models/qwen/Qwen2-7B-Instruct",
        "hf_path": "Qwen/Qwen2-7B-Instruct",
        "ms_path": "qwen/Qwen2-7B-Instruct",
    },
    {
        "model_alias": "llama3-8B-chat",
        "template": "llama3",
        "model_name_or_path": "../models/shenzhi-wang/Llama3-8B-Chinese-Chat",
        "hf_path": "shenzhi-wang/Llama3-8B-Chinese-Chat",
        "ms_path": "",
    },
]

def _model_list() -> List[dict]:
    model_list = MODEL_MAPPING
    return model_list

# TODO 配置模型MODEL_MAPPING放到配置文件,可以修改

# TODO 下载模型
