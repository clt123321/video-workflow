from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage
import base64


if __name__ == '__main__':
    # 1. 准备图片的Base64编码
    def image_to_base64(image_path):
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
        return f"data:image/{image_path.split('.')[-1]};base64,{encoded_string}"

    # 替换为你的图片路径
    image_path = "C:\\Users\clt\Pictures\猫.jpg"
    image_base64 = image_to_base64(image_path)

    # 2. 创建LangChain ChatOllama实例
    llava_chat = ChatOllama(
        model="llava",  # 确保本地已下载模型：ollama pull llava
        base_url="http://localhost:11434",
        temperature=0.7
    )

    # 3. 构建多模态消息
    messages = [
        HumanMessage(
            content=[
                {"type": "text", "text": "What is in this picture?"},
                {"type": "image_url", "image_url": {"url": image_base64}}
            ]
        )
    ]

    # 4. 发送请求并获取响应
    response = llava_chat.invoke(messages)
    print("模型回复：", response.content)