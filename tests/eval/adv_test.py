# 模型加载
from datetime import datetime
from llamafactory.chat import ChatModel
from llamafactory.extras.misc import torch_gc
from llamafactory.f1.recall_precise_rate import F1score
import json
from parse_ds.sft_prompt import sft_unit_prompt
from langchain.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tiktoken

from llamafactory.chat import ChatModel

guide_prompt = """你是一个医学专家同时精通各种医学自然语言处理技术，你的任务是从药物不良反应专家指南中提取服用

                药品名：|||drug_name||| \后不良反应的患者自我管理办法和治疗措施。
                在具体的提取任务开始前，你需要明确，自我管理办法和治疗措施的定义差别。自我管理办法主要从患者的日常出发，利用改变一些生活习惯比如作息、饮食等，来进行不良反应的干预；治疗措施通常指从用药出发，利用药物的作用干预不良反应的症状。

                不良反应相关内容：|||input||| \

                输出格式: 确保按照以下JSON格式的代码片段输出结果。
                {
                "不良反应具体类型": {
                                "管理办法":
                                "治疗措施":
                } // 注意"不良反应具体类型"字段需要替换具体的不良反应名称，而不是直接返回“不良反应具体类型”，同时所有输出均用中文表达；注意：你返回的json类型必须是上述json格式！"""


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def read_txt(file_path):
    with open(file_path, "r") as f:
        guide_txt = f.read()
    return guide_txt


def split_txt(text):
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name="gpt-3.5-turbo-1106",
        chunk_size=2000,
        chunk_overlap=20,
    )
    pages = text_splitter.split_text(text)
    texts = [Document(page_content=p) for p in pages]
    print(len(texts))
    return texts


# map-reduce
def run_guide_map_reduce(texts, drug, model: "ChatModel"):
    chunks = split_txt(texts)
    res = []
    print(res)
    for chunk in chunks:
        messages = []
        messages.append(
            {"role": "user", "content": guide_prompt.replace("|||drug_name|||", drug).replace("|||input|||", chunk.page_content)}
        )
        response = ""
        response = model.chat(messages)
        response = response[0].response_text
        res.append(response)
        # print("model answer：", response)
    print(res)
    return res


# refine
def run_refine(texts, model: "ChatModel"):

    return


# 20B=200亿token 0.01亿字（红楼梦）
print("*****************运行评估测试************************")
args = dict(
    do_sample=True,
    model_name_or_path="/mnt/windows/Users/Admin/LLM/models/qwen/Qwen1___5-14B-Chat-GPTQ-Int8",
    # adapter_name_or_path="/mnt/windows/Users/Admin/LLM/models/AI-ModelScope/sfan_gemma2b_test",
    template="qwen",  # 和训练保持一致
    # finetuning_type="lora",  # 和训练保持一致
    # quantization_bit=4,                    # 加载 4 比特量化模型
    temperature=0.95,
    top_p=0.7,
    max_new_tokens=2048,
    repetition_penalty=1.0,
    length_penalty=1.0,
    num_beams=1,
    top_k=80,
)


if __name__ == "__main__":
    # 指南提取
    texts = read_txt("tests/eval/dufa.txt")
    torch_gc()
    chat_model = ChatModel(args)
    drug_name = "度伐利尤单抗"
    run_guide_map_reduce(texts, drug_name, chat_model)
    # for new_text in chat_model.stream_chat(messages):
    #     print(new_text, end="")
    #     response += new_text
