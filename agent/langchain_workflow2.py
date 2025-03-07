import time

from agent.langchain_workflow1 import MultiModalRAG
from data.prompt.const_prompt import LLM_POLICY_GENERATION_PROMPT


class MultiModalRAG_IMAGE(MultiModalRAG):
    def ask_vlm(self, question, image_path, prompt=None):
        print(f"start to ask_vlm, question: {question}")
        # 记录开始时间
        start_time = time.time()
        # step1:使用clip对图像编码
        img_base64 = self._encode_image(image_path)
        # todo:存储图像编码
        # 计算并打印执行时间
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"finish ask_vlm, running cost time {execution_time:.6f} seconds")
        # 获取响应
        return 1

if __name__ == '__main__':
    # 首次使用需要构建知识库
    # build_image_vector_db()

    # 初始化系统
    rag_system = MultiModalRAG_IMAGE()

    # 示例查询
    prompt = LLM_POLICY_GENERATION_PROMPT
    response = rag_system.ask_vlm(
        question="why the image related to context, give me reasons?",
        image_path="../data/input/image/red_dream.png",
        prompt=None
    )