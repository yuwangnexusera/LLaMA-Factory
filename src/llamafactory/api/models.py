
from typing import  List

from ..data import Role as DataRole
from ..extras.logging import get_logger
from .common import dictify, jsonify
from .protocol import (
    Role
)



logger = get_logger(__name__)
ROLE_MAPPING = {
    Role.USER: DataRole.USER.value,
    Role.ASSISTANT: DataRole.ASSISTANT.value,
    Role.SYSTEM: DataRole.SYSTEM.value,
    Role.FUNCTION: DataRole.FUNCTION.value,
    Role.TOOL: DataRole.OBSERVATION.value,
}
MODEL_MAPPING = {
    "qwen2-7B-chat": {"model_name": "qwen2-7b-chat", "model_path": "../models/qwen/Qwen2-7B-Instruct", "template": "qwen2"}
}

def _model_list() -> List[str]:
    model_list = []
    for model_name,confs in MODEL_MAPPING.items():
        model_list.append({"model_name": model_name, "reletive_path": confs["model_path"]})
    return jsonify(model_list)
