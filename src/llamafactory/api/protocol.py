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

import time
from enum import Enum, unique
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field
from typing_extensions import Literal
@unique
class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    FUNCTION = "function"
    TOOL = "tool"


@unique
class Finish(str, Enum):
    STOP = "stop"
    LENGTH = "length"
    TOOL = "tool_calls"


class ModelCard(BaseModel):
    model_alias: str = "Qwen1.5-14B-int8"
    template: str = "qwen"
    model_name_or_path: str
    unsloth: bool = False
    description: Optional[str]


class ModelList(BaseModel):
    data: List[ModelCard] = []


class LoadModelRequest(BaseModel):
    model_alias: str = "Qwen1.5-14B-int8"
    # do_sample: bool = True
    # adapter_name_or_path: str = "output_model_dir"
    # template: str = "qwen"
    num_beams:int=3
    use_unsloth: bool = False
    temperature: float = 0.3
    top_p: float = 0.7
    max_new_tokens: int = 1024
    repetition_penalty: float = 1.0
    length_penalty: float = 1.1
    top_k :int = 80

class LoadModelRequestBody(BaseModel):
    model_name_or_path: str
    template: str
    num_beams:int 
    temperature: float 
    top_p: float
    max_new_tokens: int 
    repetition_penalty: float 
    length_penalty: float 
    top_k:int


class LoadModelResponse(BaseModel):
    status: Literal["success", "failed"]
    message: str


class SingleReportRequest(BaseModel):
    unit_name: str = "治疗用药方案"
    report: str


class SingleReportResponse(BaseModel):
    unit_json: str


class Function(BaseModel):
    name: str
    arguments: str


class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Any]


class FunctionAvailable(BaseModel):
    type: Literal["function", "code_interpreter"] = "function"
    function: Optional[FunctionDefinition] = None


class FunctionCall(BaseModel):
    id: str
    type: Literal["function"] = "function"
    function: Function


class ImageURL(BaseModel):
    url: str


class MultimodalInputItem(BaseModel):
    type: Literal["text", "image_url"]
    text: Optional[str] = None
    image_url: Optional[ImageURL] = None


class ChatMessage(BaseModel):
    role: Role
    content: Optional[Union[str, List[MultimodalInputItem]]] = None
    # tool_calls: Optional[List[FunctionCall]] = None


class ChatCompletionMessage(BaseModel):
    role: Optional[Role] = None
    content: Optional[str] = None
    tool_calls: Optional[List[FunctionCall]] = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    tools: Optional[List[FunctionAvailable]] = None
    do_sample: Optional[bool] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    n: int = 1
    max_tokens: Optional[int] = 1024
    stop: Optional[Union[str, List[str]]] = None
    stream: bool = False


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatCompletionMessage
    finish_reason: Finish


class ChatCompletionStreamResponseChoice(BaseModel):
    index: int
    delta: ChatCompletionMessage
    finish_reason: Optional[Finish] = None


class ChatCompletionResponseUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: ChatCompletionResponseUsage


class BenchmarkResponse(BaseModel):
    evaluation_criteria: Union[dict, str]
    model_correct_answer:dict= {}
    error_details : Union[dict, str]


class BenchmarkRequest(BaseModel):
    benchmark: str = "Structured medical records"
    test_unit: List[str] = ["治疗用药方案"]
    # test_prompt:str = "可不填,用默认微调的prompt"
    samples: int = 1
    temperature: float = 0.01
    top_p: float = 0.8
    max_new_tokens: int = 1024
    repetition_penalty: float = 1.2
    length_penalty: float = 1.1


class ChatCompletionStreamResponse(BaseModel):
    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionStreamResponseChoice]

class ScoreEvaluationRequest(BaseModel):
    model: str
    messages: List[str]
    max_length: Optional[int] = None


class ScoreEvaluationResponse(BaseModel):
    id: str
    object: Literal["score.evaluation"] = "score.evaluation"
    model: str
    scores: List[float]
