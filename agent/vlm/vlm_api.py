import http.client
import json

key = ""

def get_vlm_response():
    conn = http.client.HTTPSConnection("api.tutujin.com")
    payload = json.dumps({
        "model": "gpt-4-vision-preview",
        "stream": False,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "这张图片有什么"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://github.com/dianping/cat/raw/master/cat-home/src/main/webapp/images/logo/cat_logo03.png"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 400
    })
    headers = {
        'Accept': 'application/json',
        'Authorization': 'Bearer ' + key,
        'Content-Type': 'application/json'
    }
    conn.request("POST", "/v1/chat/completions", payload, headers)
    res = conn.getresponse()
    data = res.read()
    return data.decode("utf-8")


if __name__ == '__main__':
    response = get_vlm_response()
    print(response)
