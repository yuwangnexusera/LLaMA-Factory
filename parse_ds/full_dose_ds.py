from openai import OpenAI
import tiktoken
def ask_doubao(prompt):
    client = OpenAI(
        api_key="61ae527f-0de1-4ad9-8f5f-18b8ba296c1f",
        base_url="https://ark.cn-beijing.volces.com/api/v3",
    )

    # Non-streaming:
    print("----- standard request -----")
    completion = client.chat.completions.create(
        model="ep-20240716085539-pqjhg",  # your model endpoint ID
        messages=[
            {"role": "system", "content": "你的任务是按要求从给定报告中提取医学信息。"},
            {"role": "user", "content": prompt},
        ],
        stream=True,
    )
    output_llm = ""
    encoding = tiktoken.encoding_for_model(model_name="gpt-4")  # gpt-4/gpt-3.5-turbo
    token_num = len(encoding.encode(prompt))
    for chunk in completion:
        if chunk.choices[0].delta.content is not None:
            print(chunk.choices[0].delta.content, end="")
            output_llm += chunk.choices[0].delta.content
    return output_llm

def gene_prompt_extract():


    return  content