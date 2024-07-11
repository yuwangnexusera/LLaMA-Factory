from langchain.chains.summarize import load_summarize_chain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain
from langchain_text_splitters import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tiktoken
import os

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def read_txt(file_path):
    with open(file_path,'r') as f:
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
# TODO 指南切割，总结，治疗方法

if __name__ == "__main__":
    txt_str = read_txt("tests/eval/util/dufa.txt")
    num_tokens_from_string(txt_str, "cl100k_base")
    # split_guide("src/tests/eval/util/dufa.txt")
