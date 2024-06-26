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
MODEL_MAPPING = {
    "qwen2-7B-chat": {
        "hf_path": "Qwen/Qwen2-7B-Instruct",
        "ms_path": "qwen/Qwen2-7B-Instruct",
        "model_path": "../models/qwen/Qwen2-7B-Instruct",
        "template": "qwen2",
    }
}

def _model_list() -> List[dict]:
    model_list = []
    for model_name,confs in MODEL_MAPPING.items():
        model_list.append({model_name: confs})
    return model_list
