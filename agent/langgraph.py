import json

import requests

from agent.langchain_workflow1 import parse_json

if __name__ == '__main__':
    prompt = "你好"
    system_prompt = "你是一个礼仪小姐"

    url = "https://api.siliconflow.cn/v1/chat/completions"

    payload = {
        "model": "deepseek-ai/DeepSeek-R1",
        "messages": [
            {
                "role": "user",
                "content": prompt
            },
            {
                "role": "system",
                "content": prompt
            }
        ],
        "stream": False,
        "max_tokens": 512,
        "stop": None,
        "temperature": 0.7,
        "top_p": 0.7,
        "top_k": 50,
        "frequency_penalty": 0.5,
        "n": 1,
        "response_format": {"type": "text"},
        "tools": [
            {
                "type": "function",
                "function": {
                    "description": "<string>",
                    "name": "<string>",
                    "parameters": {},
                    "strict": False
                }
            }
        ]
    }
    headers = {
        "Authorization": "Bearer sk-sywelvfbeymcbfwolkwgdscrkukbsynxxkpflpriiqbhyqjw",
        "Content-Type": "application/json"
    }
    response = requests.request("POST", url, json=payload, headers=headers)
    print(response.status_code)  # == 200
    #
    #     deepseek-ai/DeepSeek-V3回复 = """
    #     {
    #   "id" : "01958126e32a20d4dd43cd0ff5aed723",
    #   "object" : "chat.completion",
    #   "created" : 1741628564,
    #   "model" : "deepseek-ai/DeepSeek-V3",
    #   "choices" : [ {
    #     "index" : 0,
    #     "message" : {
    #       "role" : "assistant",
    #       "content" : "你好！很高兴见到你。今天有什么我可以帮你的吗？"
    #     },
    #     "finish_reason" : "stop"
    #   } ],
    #   "usage" : {
    #     "prompt_tokens" : 54,
    #     "completion_tokens" : 13,
    #     "total_tokens" : 67
    #   },
    #   "system_fingerprint" : ""
    # }"""

#     deepseek-ai/DeepSeek-R1回复 = """
    #     {
    #   "id" : "01958133eb7c5e4d8a3cf5b49b1df362",
    #   "object" : "chat.completion",
    #   "created" : 1741629418,
    #   "model" : "deepseek-ai/DeepSeek-R1",
    #   "choices" : [ {
    #     "index" : 0,
    #     "message" : {
    #       "role" : "assistant",
    #       "content" : "\n\n你好！很高兴见到你，有什么我可以帮忙的吗？无论是学习、工作还是生活中的问题，都可以告诉我哦！😊"
    #     },
    #     "finish_reason" : "stop"
    #   } ],
    #   "usage" : {
    #     "prompt_tokens" : 57,
    #     "completion_tokens" : 26,
    #     "total_tokens" : 83
    #   },
    #   "system_fingerprint" : ""
    # }"""
    response_json = parse_json(response.text)
    print(response_json["choices"][0]["message"]["content"])
