def shiyingzheng_ask_llm(prompt):
    try:
        class syz_OutputFormat(BaseModel):
            jingpin: List[str] = Field(description="该药物对象每一处上下文作为列表的一项")

        completion = client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",
            messages=[
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            response_format=syz_OutputFormat,
        )
        logger.info(f"{len(prompt)}-提取中")
        # 创建聊天完成请求
        logger.info(f"{len(prompt)}-提取完成")
        output = completion.choices[0].message.parsed
        result = {
            "竞品名称": output.jingpin,
        }

    except Exception as e:
        logger.error("openai.run 异常{}".format(e))
        result = {}
    return result
