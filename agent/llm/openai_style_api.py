
import json
from openai import OpenAI

# 本地配置代理，和clash端口保持一致
import os
os.environ["http_proxy"] = "http://localhost:7897"
os.environ["https_proxy"] = "http://localhost:7897"

# 设置 OpenAI API 密钥
api_key = "xxxxxxxxxxxxxxxxxxxxx"
client = OpenAI(api_key = api_key)

def get_llm_response(
    system_prompt, prompt, json_format=True, model="gpt-4-1106-preview"
):
    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {"role": "user", "content": prompt},
    ]
    key = json.dumps([model, messages])
    logger.info(messages)
    cached_value = get_from_cache(key)
    if cached_value is not None:
        logger.info("Cache Hit")
        logger.info(cached_value)
        return cached_value

    print("Not hit cache \n", key)
    # input()

    # 尝试链接模型三次
    for _ in range(3):
        try:
            if json_format:
                completion = client.chat.completions.create(
                    model=model,
                    response_format={"type": "json_object"},
                    messages=messages,
                )
            else:
                completion = client.chat.completions.create(
                    model=model, messages=messages
                )
            response = completion.choices[0].message.content
            logger.info(response)
            save_to_cache(key, response)
            return response
        except Exception as e:
            logger.error(f"GPT Error: {e}")
            continue
    return "GPT Error"