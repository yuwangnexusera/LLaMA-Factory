# Copyright 2024 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import os
from contextlib import asynccontextmanager
from functools import partial
from typing import Optional

from typing_extensions import Annotated

# from .benchmark import ie_unit_benchmark
from ..chat import ChatModel
from ..extras.misc import torch_gc
from ..extras.packages import is_fastapi_available, is_starlette_available, is_uvicorn_available
from .chat import (
    create_chat_completion_response,
    create_score_evaluation_response,
    create_stream_chat_completion_response,
)
from .api_config import config_func
from .models import _model_list, single_report_extract
from .protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    SingleReportRequest,
    ModelList,
    SingleReportResponse,
    LoadModelRequest,
    LoadModelRequestBody,
    LoadModelResponse,
    BenchmarkRequest,
    BenchmarkResponse,
)

from .common import dictify, jsonify
import asyncio

if is_fastapi_available():
    from fastapi import Depends, FastAPI, HTTPException, status, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.security.http import HTTPAuthorizationCredentials, HTTPBearer


if is_starlette_available():
    from sse_starlette import EventSourceResponse


if is_uvicorn_available():
    import uvicorn

async def sweeper() -> None:
    while True:
        torch_gc()
        await asyncio.sleep(300)


@asynccontextmanager
async def lifespan(app: "FastAPI", chat_model: "ChatModel"):  # collects GPU memory
    if chat_model.engine_type == "huggingface":
        asyncio.create_task(sweeper())

    yield
    torch_gc()


def create_app(chat_model: "ChatModel") -> "FastAPI":
    root_path = os.environ.get("FASTAPI_ROOT_PATH", "")
    app = FastAPI(lifespan=partial(lifespan, chat_model=chat_model), root_path=root_path)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    api_key = os.environ.get("API_KEY", None)
    security = HTTPBearer(auto_error=False)

    async def verify_api_key(auth: Annotated[Optional[HTTPAuthorizationCredentials], Depends(security)]):
        if api_key and (auth is None or auth.credentials != api_key):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key.")

    @app.get(
        "/v1/models/list",
        response_model=ModelList,
        status_code=status.HTTP_200_OK,
        dependencies=[Depends(verify_api_key)],
    )
    async def list_models():
        model_list = _model_list()
        return ModelList(data=model_list)

    @app.post(
        "/v1/model/single_report",
        response_model=SingleReportResponse,
        status_code=status.HTTP_200_OK,
        dependencies=[Depends(verify_api_key)],
    )
    async def single_report(request: SingleReportRequest):
        # TODO 单篇报告提取 sft_unit_prompt
        return single_report_extract(str(request.report), chat_model)
    return app


def run_api() -> None:
    args = dict(
        do_sample=True,
        # model_name_or_path="/mnt/windows/Users/Admin/LLM/models/Shanghai_AI_Laboratory/internlm2_5-7b-chat",
        model_name_or_path="/app/data/LLM/models/qwen/Qwen2___5-7B-Instruct", # 原始模型权重
        # adapter_name_or_path="/mnt/windows/Users/Admin/LLM/models/Shanghai_AI_Laboratory/susu_internlm2_5_vt_1011",  # 加载之前保存的 LoRA 适配器
        adapter_name_or_path="/app/data/LLM/models/qwen/SS_Qwen2_5_7B_1020",  # 微调之后的 LoRA 适配器文件（需要替换的）
        # template="intern2",  # 和训练保持一致
        template="qwen",  # 和训练保持一致
        finetuning_type="lora",  # 和训练保持一致
        # quantization_bit=4,
        temperature=0.01,
        max_new_tokens=1024,
        # repetition_penalty=1.0,
        # length_penalty=1.1,
        num_beams=1,
    )
    chat_model = ChatModel(args)
    app = create_app(chat_model)
    api_host = os.environ.get("API_HOST", "0.0.0.0")
    api_port = int(os.environ.get("API_PORT", "8008"))
    print("Visit http://localhost:{}/docs for API document.".format(api_port))
    uvicorn.run(app, host=api_host, port=api_port,timeout_keep_alive=300,timeout_graceful_shutdown=60)
