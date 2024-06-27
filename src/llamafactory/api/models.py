from typing import List

from ..data import Role as DataRole
from ..extras.logging import get_logger
from .common import dictify, jsonify
from .protocol import Role, ModelList, DownloadModelRequest, DownloadModelResponse
import pandas

import asyncio
import subprocess
from modelscope import snapshot_download
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
model_conf_path = "api/config/model_conf.json"
MODEL_MAPPING = pandas.read_json(model_conf_path, orient="records").to_dict(orient="records")


def _model_list() -> List[dict]:
    model_list = MODEL_MAPPING
    return model_list


# TODO 下载模型
async def async_run_command(command):
    """异步执行命令行命令"""
    process = await asyncio.create_subprocess_shell(command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    stdout, stderr = await process.communicate()
    if process.returncode == 0:
        print("Download completed successfully!")
    else:
        print(f"Error in download: {stderr.decode()}")

    return "done"


async def download_model_from_modelscope(model_name, cache_dir="../models/"):
    """从modelscope下载模型"""
    try:
        model_dir = snapshot_download(model_name, cache_dir=cache_dir)
        print(f"Model downloaded to {model_dir}")
        return "done"
    except Exception as e:
        print(f"Error downloading from Modelscope: {str(e)}")
        return "failed"


def update_configuration(request):
    """更新配置文件"""
    try:
        if request.origin_platform=="modelscope":
            ms_path = request.model_name_or_path
            hf_path = ""
        elif request.origin_platform=="huggingface":
            ms_path = ""
            hf_path = request.model_name_or_path
        MODEL_MAPPING.append(
            {
                "model_alias": request.model_alias,
                "template": request.template,
                "model_name_or_path": '../models/' + request.model_name_or_path,
                "hf_path": hf_path,
                "ms_path": ms_path,
            }
        )
        MODEL_MAPPING.to_json(model_conf_path, orient="records", lines=True)
        print("Configuration updated successfully!")
    except Exception as e:
        print(f"Error updating configuration: {str(e)}")


async def download_model(request: DownloadModelRequest):
    """下载模型的主接口函数"""
    if request.origin_platform == "modelscope":
        status = await download_model_from_modelscope(request.model_name_or_path)
    elif request.origin_platform == "huggingface":
        command = f"python utils/hf_download.py --model {request.model_name_or_path} --token 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 --save_dir ../models/"
        status = await async_run_command(command)
    else:
        return DownloadModelResponse(msg="origin_platform参数错误")
    if status == "done":
        return DownloadModelResponse(msg=update_configuration(request))
    else:
        return DownloadModelResponse(msg="下载失败")
