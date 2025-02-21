from io import BytesIO

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings  # 更新后的导入方式
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage
from PIL import Image
import base64


# ================= 更新后的知识库构建 =================
def build_knowledge_base():
    # 加载《奥德赛》文本
    loader = TextLoader("../data/input/text/odyssey.txt", encoding="utf-8")
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
    vector_db.save_local("odyssey_faiss_updated")


# ================= 更新后的RAG系统 =================
class UpdatedMultiModalRAG:
    def __init__(self):
        # 加载新版嵌入模型
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cuda"},  # 启用GPU加速
            encode_kwargs={"normalize_embeddings": True}
        )

        # 加载向量数据库
        self.vector_db = FAISS.load_local(
            "odyssey_faiss_updated",
            self.embeddings,
            allow_dangerous_deserialization=True  # 需要安全确认
        )

        # 初始化LLaVA
        self.llava = ChatOllama(
            model="llava",  # 确保本地已下载模型：ollama pull llava
            base_url="http://localhost:11434",
            temperature=0.7
        )

    def _encode_image(self, path):
        with Image.open(path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            buffered = BytesIO()
            img.save(buffered, format="JPEG")
            return f"data:image/jpeg;base64,{base64.b64encode(buffered.getvalue()).decode()}"

    def ask(self, question, image_path):
        # 图像编码
        img_base64 = self._encode_image(image_path)

        # 知识检索（增强top3上下文）
        context = "\n".join([
            doc.page_content
            for doc in self.vector_db.similarity_search(question, k=3)
        ])

        print(f"knowledge retrival from vector_db is: {context}")

        # 构建增强提示
        prompt = f"""基于以下《奥德赛》知识片段和图片内容回答问题：

        相关文本：
        {context}

        问题：{question}
        请结合图文信息详细回答："""

        # 构建多模态消息
        messages = [HumanMessage(content=[
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": img_base64}}
        ])]

        # 获取响应
        return self.llava.invoke(messages).content


# ================= 使用示例 =================
if __name__ == "__main__":
    # 首次使用需要构建知识库
    # build_knowledge_base()

    # 初始化系统
    rag_system = UpdatedMultiModalRAG()

    # 示例查询
    response = rag_system.ask(
        question="图片中的场景与奥德修斯遇到的哪个神话生物最相关？",
        image_path="../data/input/image/猫.jpg"
    )
    print("llava response: ", response)
