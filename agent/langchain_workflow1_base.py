import time
import logging
from io import BytesIO

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings  # 更新后的导入方式
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage
from PIL import Image
import base64

from data.prompt.const_prompt import TEST_PROMPT, DEFAULT_PROMPT, DESCRIBE_PROMPT, DESCRIBE_PROMPT_BASE

logger = logging.getLogger(__name__)

# ================= 文本向量知识库构建 =================
def build_text_vector_db():
    logger.info("start to build_text_vector_db")
    # 记录开始时间
    start_time = time.time()
    # 加载文本
    loader = TextLoader("../data/input/text/red_dream.txt", encoding="utf-8")
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
        self.vector_db = FAISS.load_local(
            ".\\agent\\red_dream",
            self.embeddings,
            allow_dangerous_deserialization=True  # 需要安全确认
        )

        # 初始化LLaVA
        self.llava = ChatOllama(
            model="llava",  # 确保本地已下载模型：ollama pull llava
            base_url="http://localhost:11434",
            temperature=0
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
        # step1:图像编码
        img_base64 = self._encode_image(image_path)

        # step2:调用vlm获取图片文本描述
        message_image_des = [
            HumanMessage(
                content=[
                    {"type": "text", "text": DESCRIBE_PROMPT},
                    {"type": "image_url", "image_url": {"url": img_base64}}
                ]
            )
        ]
        image_description = self.llava.invoke(message_image_des).content
        logger.info(f"image_description:{image_description}")

        # step3:知识检索（增强top3上下文）
        context = "\n".join([
            doc.page_content
            for doc in self.vector_db.similarity_search(image_description, k=3)
        ])
        logger.info(f"knowledge retrieval from vector_db is: {context}")

        # step4:根据召回，构建提示词，并回答相关问题
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


# ================= 使用示例 =================
if __name__ == "__main__":
    # 首次使用需要构建知识库
    # build_text_vector_db()

    # 初始化系统
    rag_system = MultiModalRAG()

    # 示例查询
    prompt = TEST_PROMPT
    response = rag_system.ask_vlm(
        question="why the image related to context, give me reasons?",
        image_path="../data/input/image/red_dream.png",
        prompt=None
    )
