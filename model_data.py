[
    {
        provider: "OpenAI",
        monetary: "USD",
        uri: "https://openai.com/api/pricing/",
        models: [
            {name: "GPT-4o", inputPrice: 5.0, outputPrice: 15.0},
            {name: "GPT-4 (8K)", inputPrice: 30.0, outputPrice: 60.0},
            {name: "GPT-4 Turbo", inputPrice: 10.0, outputPrice: 30.0},
            {name: "GPT-3.5-turbo", inputPrice: 0.5, outputPrice: 1.5},
        ],
    },
    {
        provider: "Baidu",
        monetary: "RMB",
        uri: "https://cloud.baidu.com/doc/WENXINWORKSHOP/s/hlrk4akp7",
        models: [
            {name: "ERNIE 4.0 Turbo", inputPrice: 30, outputPrice: 60},
            {name: "ERNIE 4", inputPrice: 40, outputPrice: 120},
            {name: "ERNIE 3.5", inputPrice: 4, outputPrice: 12},
            {name: "ERNIE 3.5-128k", inputPrice: 8, outputPrice: 24},
            {name: "ERNIE Speed(Lite)", inputPrice: 0, outputPrice: 0},
        ],
    },
    {
        provider: "腾讯",
        monetary: "RMB",
        uri: "https://cloud.tencent.com/document/product/1729/97731",
        models: [
            {name: "hunyuan-pro", inputPrice: 30, outputPrice: 100},
            {name: "hunyuan-standard", inputPrice: 4.5, outputPrice: 5},
            {name: "hunyuan-standard-256k", inputPrice: 15, outputPrice: 60},
            {name: "hunyuan-lite", inputPrice: 0, outputPrice: 0},
        ],
    },
    {
        provider: "通义千问",
        monetary: "RMB",
        uri: "https://www.aliyun.com/product/bailian#price",
        models: [
            {name: "qwen-max", inputPrice: 40, outputPrice: 120},
            {name: "qwen-max-1201", inputPrice: 120, outputPrice: 120},
            {name: "qwen-plus", inputPrice: 4, outputPrice: 12},
            {name: "qwen-turbo", inputPrice: 2, outputPrice: 6},
            {name: "qwen-long", inputPrice: 0.5, outputPrice: 2},
        ],
    },
    {
        provider: "智普",
        monetary: "RMB",
        uri: "",
        models: [{name: "GLM-4-0520", inputPrice: 100, outputPrice: 100}],
    },
    {
        provider: "01.ai",
        monetary: "RMB",
        uri: "https://platform.lingyiwanwu.com/docs#%E4%BA%A7%E5%93%81%E5%AE%9A%E4%BB%B7",
        models: [{name: "Yi-Large", inputPrice: 20, outputPrice: 20}, {name: "yi-large-turbo", inputPrice: 12, outputPrice: 12}],
    },
    {
        provider: "百川",
        monetary: "RMB",
        uri: "https://platform.baichuan-ai.com/price",
        models: [
            {name: "Baichuan4", inputPrice: 100, outputPrice: 100},
            {name: "Baichuan3-Turbo", inputPrice: 12, outputPrice: 12},
            {name: "Baichuan3-Turbo-128k", inputPrice: 24, outputPrice: 24},
        ],
    },
    {
        provider: "豆包",
        monetary: "RMB",
        uri: "",
        models: [
            {name: "Doubao-pro", inputPrice: 0.8, outputPrice: 2},
            {name: "Doubao-pro-128k", inputPrice: 5, outputPrice: 9},
            {name: "Doubao-lite", inputPrice: 0.3, outputPrice: 0.6},
            {name: "Doubao-lite-128k", inputPrice: 0.8, outputPrice: 1},
        ],
    },
    {
        provider: "Anthropic",
        monetary: "USD",
        uri: "https://www.anthropic.com/pricing",
        models: [
            {name: "Claude 3 (Opus)", inputPrice: 15.0, outputPrice: 75.0},
            {name: "Claude 3.5 (Sonnet)", inputPrice: 3.0, outputPrice: 15.0},
            {name: "Claude 3 (Haiku)", inputPrice: 0.25, outputPrice: 1.25},
        ],
    },
    {
        provider: "Google",
        monetary: "USD",
        uri: "https://ai.google.dev/pricing?hl=en",
        models: [
            {name: "Gemini 1.5 Pro", inputPrice: 3.5, outputPrice: 7.0},
            {name: "Gemini 1.5 Flash", inputPrice: 0.35, outputPrice: 0.7},
        ],
    },
    {
        provider: "Cohere",
        monetary: "USD",
        uri: "https://cohere.com/pricing",
        models: [
            {name: "Command R+", inputPrice: 3.0, outputPrice: 15.0},
            {name: "Command R", inputPrice: 0.5, outputPrice: 1.5},
        ],
    },
    {
        provider: "Mistral",
        monetary: "USD",
        uri: "https://docs.mistral.ai/platform/pricing",
        models: [
            {name: "mistral-large-2402", inputPrice: 4.0, outputPrice: 12.0},
            {name: "codestral-2405", inputPrice: 1.0, outputPrice: 3.0},
            {name: "Mixtral 8x22B", inputPrice: 2.0, outputPrice: 6.0},
            {name: "Mixtral 8x7B", inputPrice: 0.7, outputPrice: 0.7},
        ],
    },
    {
        provider: "Deepspeek",
        monetary: "USD",
        uri: "https://platform.deepseek.com/api-docs/pricing/",
        models: [
            {name: "deepseek-chat", inputPrice: 0.14, outputPrice: 0.28},
            {name: "deepseek-coder", inputPrice: 0.14, outputPrice: 0.28},
        ],
    },
    {
        provider: "Groq",
        monetary: "USD",
        uri: "https://wow.groq.com/",
        models: [
            {name: "Llama 3 70b", inputPrice: 0.59, outputPrice: 0.79},
            {name: "Mixtral 8x7B", inputPrice: 0.24, outputPrice: 0.24},
        ],
    },
]
