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

import os
from contextlib import asynccontextmanager
from typing import Optional

from typing_extensions import Annotated
from .benchmark import ie_unit_benchmark
from ..chat import ChatModel
from ..extras.misc import torch_gc
from ..extras.packages import is_fastapi_available, is_starlette_available, is_uvicorn_available
from .chat import (
    create_chat_completion_response,
    create_score_evaluation_response,
    create_stream_chat_completion_response,
)
from .models import _model_list
from .protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ModelCard,
    ModelList,
    ScoreEvaluationRequest,
    ScoreEvaluationResponse,
    LoadModelRequest,
    LoadModelResponse,
    BenchmarkRequest,
    BenchmarkResponse,
)
from .common import dictify, jsonify

if is_fastapi_available():
    from fastapi import Depends, FastAPI, HTTPException, status
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.security.http import HTTPAuthorizationCredentials, HTTPBearer


if is_starlette_available():
    from sse_starlette import EventSourceResponse


if is_uvicorn_available():
    import uvicorn


@asynccontextmanager
async def lifespan(app: "FastAPI"):  # collects GPU memory
    yield
    torch_gc()


def create_app() -> "FastAPI":
    app = FastAPI(lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    api_key = os.environ.get("API_KEY")
    security = HTTPBearer(auto_error=False)

    async def verify_api_key(auth: Annotated[Optional[HTTPAuthorizationCredentials], Depends(security)]):
        if api_key and (auth is None or auth.credentials != api_key):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key.")

    @app.get(
        "/v1/models",
        response_model=ModelList,
        status_code=status.HTTP_200_OK,
        dependencies=[Depends(verify_api_key)],
    )
    async def list_models():
        model_list = _model_list()
        return ModelList(data=model_list)

    @app.post(
        "/v1/model/load",
        response_model=LoadModelResponse,
        status_code=status.HTTP_200_OK,
        dependencies=[Depends(verify_api_key)],
    )
    async def load_model(load_args: LoadModelRequest):
        torch_gc()
        try:
            app.state.chat_model = ChatModel(dictify(load_args))
            return LoadModelResponse(status="success", message=f"{load_args.model_name_or_path}Model loaded")
        except Exception as err:
            return LoadModelResponse(status="failed", message=str(err))

    # benchmark接口 TODO 错误原因，模型答案，标准答案
    @app.post(
        "/v1/model/benchmark",
        description="benchmark，模型答案，标准答案",
        response_model=BenchmarkResponse,
        status_code=status.HTTP_200_OK,
        dependencies=[Depends(verify_api_key)],
    )
    async def benchmark_test(request: BenchmarkRequest):

        return ie_unit_benchmark(request, app.state.chat_model)

    @app.post(
        "/v1/model/chat",
        response_model=ChatCompletionResponse,
        status_code=status.HTTP_200_OK,
        dependencies=[Depends(verify_api_key)],
    )
    async def create_chat_completion(request: ChatCompletionRequest):
        try:
            if not app.state.chat_model.engine.can_generate:
                raise HTTPException(status_code=status.HTTP_405_METHOD_NOT_ALLOWED, detail="Not allowed")

            if request.stream:
                generate = create_stream_chat_completion_response(request, app.state.chat_model)
                return EventSourceResponse(generate, media_type="text/event-stream")
            else:
                return await create_chat_completion_response(request, app.state.chat_model)
        except Exception as err:
            return ChatCompletionResponse(id=err+"模型可能未加载.....",)

    # @app.post(
    #     "/v1/score/evaluation",
    #     response_model=ScoreEvaluationResponse,
    #     status_code=status.HTTP_200_OK,
    #     dependencies=[Depends(verify_api_key)],
    # )
    # async def create_score_evaluation(request: ScoreEvaluationRequest):
    #     if app.state.chat_model.engine.can_generate:
    #         raise HTTPException(status_code=status.HTTP_405_METHOD_NOT_ALLOWED, detail="Not allowed")

    #     return await create_score_evaluation_response(request, app.state.chat_model)

    return app


def run_api() -> None:
    app = create_app()
    api_host = os.environ.get("API_HOST", "0.0.0.0")
    api_port = int(os.environ.get("API_PORT", "8000"))
    print("Visit http://localhost:{}/docs for API document.".format(api_port))
    uvicorn.run(app, host=api_host, port=api_port)
