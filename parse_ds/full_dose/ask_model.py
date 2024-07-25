from openai import OpenAI
import logging 
import sys
sys.path.append(".")
from env.env_llm import OPENAI
logging.basicConfig(level=logging.INFO,format="%(asctime)s - %(levelname)s - %(message)s")
class AskModel:
    def __init__(self,model_name) -> None:
        assert model_name in ["doubao","gpt-4"]
        self.model_name = model_name

    def ask_doubao(self,prompt):
        client = OpenAI(
            api_key="61ae527f-0de1-4ad9-8f5f-18b8ba296c1f",
            base_url="https://ark.cn-beijing.volces.com/api/v3",
        )

        completion = client.chat.completions.create(
            model="ep-20240715085700-26868",
            messages=[
                # {"role": "system", "content": "根据用户要求提取信息"},
                {"role": "user", "content": prompt},
            ],
            stream=True,
            temperature=0.1,
            top_p=0.8
        )
        output_llm = ""
        for chunk in completion:
            if chunk.choices[0].delta.content is not None:
                print(chunk.choices[0].delta.content, end="")
                output_llm += chunk.choices[0].delta.content
        # completion.close()
        return output_llm
    def ask_llm(self,prompt):

        client = OpenAI(
            # This is the default and can be omitted
            api_key=OPENAI["OPENAI_API_KEY"],
            base_url=OPENAI["OPENAI_API_BASE"],
        )
        model_name = "gpt-4o"
        client._client
        try:
            # 创建聊天完成请求
            completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model=model_name,
                temperature=0.01,
                stream=True,
                
                # timeout=120
            )
            # breakpoint()
            print(f"answer by :{model_name}")
            output = ""
            for chunk in completion:
                if chunk.choices[0].delta.content is not None:
                    print(chunk.choices[0].delta.content, end="")
                    output += chunk.choices[0].delta.content
            # output = completion.choices[0].message.content
        except Exception as e:
            print(f"@ask_llm :{e}")
            output = ""  # 设置为 None，表示提取失败
        return output
