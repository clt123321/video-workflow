import time
import logging
from io import BytesIO
import re
from typing import Dict, Union

from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings  # 更新后的导入方式
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage
from PIL import Image
import base64

from config import PROJECT_ROOT
from data.prompt.const_prompt import VLM_TEST_PROMPT, DEFAULT_PROMPT, VLM_DESCRIBE_PROMPT, VLM_DESCRIBE_PROMPT_BASE

logger = logging.getLogger(__name__)


# ================= 文本向量知识库构建 =================
def build_text_vector_db():
    logger.info("start to build_text_vector_db")
    # 记录开始时间
    start_time = time.time()
    # 加载文本
    loader = TextLoader(PROJECT_ROOT + "/data/input/text/red_dream.txt", encoding="utf-8")
    documents = loader.load()

    # 文本分块
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80,
        separators=["\n\n", "\n", "。", "！", "？"]
    )
    chunks = text_splitter.split_documents(documents)

    # 使用新版嵌入模型（添加encode_kwargs参数）
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs={"normalize_embeddings": True}  # 新增推荐参数
    )

    # 创建向量数据库
    vector_db = FAISS.from_documents(chunks, embeddings)
    vector_db.save_local("red_dream")

    # 计算并打印执行时间
    end_time = time.time()
    execution_time = end_time - start_time
    logger.info(f"finish build_text_vector_db, cost time {execution_time:.6f} seconds")


def parse_deepseek_response(response: Union[str, dict]) -> Dict[str, str]:
    """
    解析DeepSeek响应中带<think>标签的结构化内容
    支持两种输入格式：
    1. 直接字符串："<think>思考过程</think> 最终回答"
    2. LangChain响应对象：response.content包含上述字符串

    返回字典格式：
    {
        "thought": "思考过程内容",
        "answer": "最终回答内容",
        "raw": "原始完整内容"
    }
    """
    # 统一提取文本内容
    text = response.content if hasattr(response, 'content') else response

    # 使用非贪婪模式匹配思考内容
    thought_pattern = re.compile(r'<think>(.*?)</think>', re.DOTALL)
    thought_match = thought_pattern.search(text)

    # 提取思考内容
    thought = thought_match.group(1).strip() if thought_match else ""

    # 提取最终回答（移除思考部分）
    answer = thought_pattern.sub('', text).strip()

    # 处理多轮对话场景（如果有多个think标签）
    if thought and not answer:
        answer = text.split('</think>')[-1].strip()

    return {
        "thought": thought,
        "answer": answer,
        "raw": text
    }


# ================= 多模态RAG系统，向量数据库内为文本信息 =================
class MultiModalRAG:
    def __init__(self):
        logger.info("init MultiModalRAG")
        # 加载新版嵌入模型 ps:需要开vpn
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cuda"},  # 启用GPU加速
            encode_kwargs={"normalize_embeddings": True}
        )

        # 加载向量数据库
        vector_db_path = str(PROJECT_ROOT) + "/agent/red_dream"
        self.vector_db = FAISS.load_local(
            vector_db_path,
            self.embeddings,
            allow_dangerous_deserialization=True  # 需要安全确认
        )

        # 初始化LLaVA
        self.llava = ChatOllama(
            model="llava",  # 确保本地已下载模型：ollama run llava
            base_url="http://localhost:11434",
            temperature=0.6,
            max_tokens=512
        )

        self.deepseek_r1 = ChatOllama(
            model="deepseek-r1:7b",  # 确保模型名称与本地一致
            base_url="http://localhost:11434",  # 直连默认端口
            temperature=0.6,
            max_tokens=512  # 回答字数限制,1个token ≈ 3-4个英文字符 或 1-2个中文字符
        )

    def _encode_image(self, path):
        with Image.open(path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            buffered = BytesIO()
            img.save(buffered, format="JPEG")
            return f"data:image/jpeg;base64,{base64.b64encode(buffered.getvalue()).decode()}"

    def ask_vlm(self, question, image_path, prompt=None):
        logger.info(f"base class, start to ask_vlm, question: {question}")
        logger.info(f"start to ask_vlm, question: {question}")
        # 记录开始时间
        start_time = time.time()
        # step1:调用vlm获取图片文本描述
        img_base64 = self._encode_image(image_path)
        message_image_des = [
            HumanMessage(
                content=[
                    {"type": "text", "text": VLM_DESCRIBE_PROMPT_BASE},
                    {"type": "image_url", "image_url": {"url": img_base64}}
                ]
            )
        ]
        image_description = self.llava.invoke(message_image_des).content
        logger.info(f"image_description:{image_description}")

        # step2:知识检索（增强top3上下文）
        context = "\n".join([
            doc.page_content
            for doc in self.vector_db.similarity_search(image_description, k=3)
        ])
        logger.info(f"knowledge retrieval from vector_db is: {context}")

        # step3:根据召回，构建提示词，并回答相关问题
        # default_prompt = DEFAULT_PROMPT
        # final_prompt = (prompt or default_prompt).format(
        #     context=context,
        #     question=question
        # )
        # logger.info(f"final_prompt: {final_prompt}")
        # messages2 = [HumanMessage(content=[
        #     {"type": "text", "text": final_prompt},
        #     {"type": "image_url", "image_url": {"url": img_base64}}
        # ])]
        # res = self.llava.invoke(messages2).content
        # logger.info("llava response to question: ", res)
        # 计算并打印执行时间
        end_time = time.time()
        execution_time = end_time - start_time
        logger.info(f"finish ask_vlm, running cost time {execution_time:.6f} seconds")
        # 获取响应
        return image_description

    # SwiftSage；Agent 分为快速直觉系统（S1）和慢速规划系统（S2）。S1 负责快速生成初步计划，S2 通过反思和外部工具验证计划的可行性。
    def give_answer_method1(self, video_id, question, options):
        formatted_question = (
                f"Here is the question: {question}\n"
                + "Here are the choices: "
                + " ".join([f"{i}. {ans}" for i, ans in enumerate(options)])
        )

        logger.info(f"formatted_question = {formatted_question}")
        pass

    def ask_llm(self, question, system_prompt=None):
        # 记录开始时间
        start_time = time.time()
        # 构建提示模板
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_prompt),  # 系统提示词
            ("human", "{input}")
        ])

        # 创建执行链
        chain = prompt_template | self.deepseek_r1  # 管道操作符，表示将前一个组件的输出作为后一个组件的输入
        response = chain.invoke({
            "input": question  # 输入问题
        })
        parsed_response = parse_deepseek_response(response)

        # 计算并打印执行时间
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"finish ask_llm, running cost time {execution_time:.6f} seconds")
        return parsed_response


# ================= 使用示例 =================
if __name__ == "__main__":
    # 首次使用需要构建知识库
    # build_text_vector_db()

    # 初始化系统
    rag_system = MultiModalRAG()

    # 示例查询
    # response = rag_system.ask_vlm(
    #     question="why the image related to context, give me reasons?",
    #     image_path="../data/input/image/red_dream.png",
    #     prompt=None
    # )

    question = "怎么写一个线程池，要求逻辑清晰"
    parsed_response = rag_system.ask_llm(question=question, system_prompt="you are a helpful ai assistant")

    print("思考过程：\n", parsed_response["thought"])
    print("回答：\n", parsed_response["answer"])
