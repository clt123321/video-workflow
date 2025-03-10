import os
import time
import logging
from io import BytesIO
import random
import re
import json
import numpy as np
from typing import Dict, Union

from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings  # 更新后的导入方式
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document
from PIL import Image
import base64

from config import PROJECT_ROOT
from data.prompt.const_prompt import LLM_DEFAULT_SYSTEM_PROMPT, LLM_GIVE_ANSER_BY_CAPTION_PROMPT, LLM_SELF_EVAL_PROMPT, \
    LLM_SAMPLE_FRAME_WITH_CONTEXT_PROMPT

logger = logging.getLogger(__name__)


# ================= 文本向量知识库构建,测试加载长文本 =================
def build_text_vector_db():
    logger.info("start to build_text_vector_db")
    # 记录开始时间
    start_time = time.time()
    # 加载长文本《红楼梦》
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


def clean_vector_db(vector_db):
    logger.info(vector_db.index.ntotal)
    vector_db.index.reset()  # 清空索引
    vector_db.docstore = {}  # 清空文档存储
    vector_db.index_to_docstore_id = {}  # 清空ID映射
    logger.info(vector_db.index.ntotal)


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
def parsed_answer_s1(param):
    # 匹配被 ```json 包裹的 JSON 内容
    json_match = re.search(r'```json(.*?)```', param, re.DOTALL)
    if not json_match:
        return None

    json_str = json_match.group(1).strip()
    try:
        data = json.loads(json_str)
        frame_id = data.get("frame_id")
        vlm_prompt = data.get("vlm_prompt")
        # 确保两个字段都存在且不为空
        if frame_id is not None and vlm_prompt is not None:
            return frame_id, vlm_prompt
        else:
            return None
    except (json.JSONDecodeError, AttributeError):
        return None


def get_by_frame_id(frame_id):
    pass


def save_to_vector_db(fact_vector_db, sampled_caps):
    pass


def get_image_path_by_frame_id(video_id, frame_id):
    return os.path.join(
        PROJECT_ROOT, "data", "output", "extracted_frames", str(video_id), f"frame_{frame_id:05d}.jpg"
    )


def parsed_answer_s2(answer_str):
    answer = parse_text_find_number(answer_str)
    return answer


def parse_json(text):
    try:
        # First, try to directly parse the text as JSON
        return json.loads(text)
    except json.JSONDecodeError:
        # If direct parsing fails, use regex to extract JSON
        json_pattern = r"\{.*?\}|\[.*?\]"  # Pattern for JSON objects and arrays

        matches = re.findall(json_pattern, text, re.DOTALL)
        for match in matches:
            try:
                match = match.replace("'", '"')
                return json.loads(match)
            except json.JSONDecodeError:
                continue

        # If no JSON structure is found
        print("No valid JSON found in the text.")
        return None


def parse_text_find_number(text):
    item = parse_json(text)
    try:
        match = int(item["final_answer"])
        if match in range(-1, 5):
            return match
        else:
            return random.randint(0, 4)
    except Exception as e:
        logger.error(f"Answer Parsing Error: {e}")
        return -1


def parse_self_eval_response_find_confidence(text):
    item = parse_json(text)
    try:
        match = int(item["confidence"])
        if match in range(1, 4):
            return match
        else:
            return random.randint(1, 3)
    except Exception as e:
        logger.error(f"Confidence Parsing Error: {e}")
        return 1


def get_previous_information(inference_path):
    pass


def parse_self_eval_response(parsed_self_eval_response):
    pass


def split_image_description(image_description):
    pass


class MultiModalRAG:
    def __init__(self):
        logger.info("init MultiModalRAG")
        # 加载文本embedding模型 ps:需要开vpn
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cuda"},  # 启用GPU加速
            encode_kwargs={"normalize_embeddings": True}
        )

        # 初始化事实向量数据库
        self.fact_vector_db = self.init_vector_db("fact_vector_db")

        # 初始化推测向量数据库
        self.spec_vector_db = self.init_vector_db("spec_vector_db")

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

    def init_vector_db(self, db_name):
        vector_db_path = PROJECT_ROOT + "/agent/" + db_name
        if not os.path.exists(vector_db_path):
            # 创建空数据库
            empty_db = FAISS.from_documents(
                documents=[Document(page_content="")],  # 使用空内容初始化
                embedding=self.embeddings
            )
            # 删除初始化的空内容
            empty_db.delete([empty_db.index_to_docstore_id[0]])
            # 确保目录存在
            os.makedirs(os.path.dirname(vector_db_path), exist_ok=True)
            # 保存空数据库
            empty_db.save_local(vector_db_path)
        return FAISS.load_local(
            vector_db_path,
            self.embeddings,
            allow_dangerous_deserialization=True  # 需要安全确认
        )

    def _encode_image(self, path):
        with Image.open(path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            buffered = BytesIO()
            img.save(buffered, format="JPEG")
            return f"data:image/jpeg;base64,{base64.b64encode(buffered.getvalue()).decode()}"

    def ask_vlm(self, image_path, prompt=None):
        logger.info(f"start to ask_vlm, prompt: {prompt}")
        # 记录开始时间
        start_time = time.time()
        # 调用vlm获取图片文本描述
        img_base64 = self._encode_image(image_path)
        message_image_des = [
            HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": img_base64}}
                ]
            )
        ]
        image_description = self.llava.invoke(message_image_des).content
        logger.info(f"image_description:{image_description}")

        # 计算并打印执行时间
        end_time = time.time()
        execution_time = end_time - start_time
        logger.info(f"finish ask_vlm, running cost time {execution_time:.6f} seconds")

        return image_description

    def retrieve_from_vector_db(self, vector_db, text, k):
        # 知识检索（增强top3上下文）
        context = "\n".join([
            doc.page_content
            for doc in vector_db.similarity_search(text, k=k)
        ])
        logger.info(f"knowledge retrieval from vector_db is: {context}")
        return context

    def ask_llm(self, prompt, system_prompt=None):
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
            "input": prompt  # 输入问题
        })
        logger.info(f"llm_response: {response}")
        parsed_response = parse_deepseek_response(response)
        # 计算并打印执行时间
        end_time = time.time()
        execution_time = end_time - start_time
        logger.info(f"finish ask_llm, running cost time {execution_time:.6f} seconds")
        return parsed_response

    def read_caption(self, captions, sample_idx):
        video_caption = {}
        for idx in sample_idx:
            video_caption[f"frame {idx}"] = captions[idx - 1]
        return video_caption

    # 解析回答文本，找到"final_answer"标签后的数字，-1表示不确定
    def parse_text_find_number(self, text):
        try:
            match = 1
            if match in range(-1, 5):
                return match
            else:
                return random.randint(0, 4)
        except Exception as e:
            logger.error(f"Answer Parsing Error: {e}")
            return -1

    def ask_llm_sample_frame_with_context(self, question, context, num_frames):
        prompt = LLM_SAMPLE_FRAME_WITH_CONTEXT_PROMPT.format(
            num_frames=num_frames,
            context=context,
            question=question,
            answer_format={"frame_id": "xxx", "vlm_prompt": "xxx"}
        )
        response = self.ask_llm(prompt=prompt, system_prompt=LLM_DEFAULT_SYSTEM_PROMPT)
        return prompt, response

    def ask_llm_answer_with_context(self, question, caption, num_frames):
        prompt = LLM_GIVE_ANSER_BY_CAPTION_PROMPT.format(
            num_frames=num_frames,
            caption=caption,
            question=question,
            answer_format={"final_answer": "xxx"}
        )
        response = self.ask_llm(prompt=prompt, system_prompt=LLM_DEFAULT_SYSTEM_PROMPT)
        return prompt, response

    def self_eval(self, previous_information, inference_path, answer):
        self_eval_prompt = LLM_SELF_EVAL_PROMPT.format(
            previous_information=previous_information,
            answer=answer,
            confidence_format={"confidence": "xxx"}
        )
        response = self.ask_llm(prompt=self_eval_prompt, system_prompt=LLM_DEFAULT_SYSTEM_PROMPT)
        return response

    # SwiftSage；Agent 分为快速直觉系统（S1）和慢速规划系统（S2）。S1 负责快速生成初步计划，S2 通过反思和外部工具验证计划的可行性。
    # SwiftSage_RAG_VQA
    def get_answer(self, video_id, formatted_question, caps):
        # step0：预处理，均匀采样num帧，调用vlm获取文本caption
        num_frames = len(caps)
        sample_idx = np.linspace(1, num_frames, num=6, dtype=int).tolist()
        sampled_caps = self.read_caption(caps, sample_idx)
        count_frame = len(sample_idx)

        i = 0
        max_round = 1
        score_flag = False
        final_answer = None
        while i < max_round:
            # # step1:s1阶段：调用小模型，生成采样策略和vlm提示词
            # previous_prompt, s2_parsed_response = self.ask_llm_sample_frame_with_context(
            #     formatted_question, sampled_caps, num_frames
            # )
            # thought = s2_parsed_response["thought"]
            # frame_id, vlm_prompt = parsed_answer_s1(s2_parsed_response["answer"])
            # # save_to_vector_db(self.fact_vector_db, sampled_caps)  # 存储相关事实和推测
            # # save_to_vector_db(self.spec_vector_db, thought)
            # fps = 30
            # # step2:Vlm根据提示词，在指定帧分析图片，获取（实体清单，动态特征，目的分析）
            # image_path = get_image_path_by_frame_id(video_id, (frame_id - 1) * fps)
            # image_description = self.ask_vlm(image_path=image_path, prompt=vlm_prompt)
            # # fact, spec = split_image_description(image_description)  # 解析事实和推理
            # # save_to_vector_db(self.fact_vector_db, fact)  # 存储相关事实和推测
            # # save_to_vector_db(self.spec_vector_db, spec)

            # step3:S2慢速推理（根据事实产生答案和分析过程）; llm预测答案，生成推理过程
            previous_prompt, s2_parsed_response = self.ask_llm_answer_with_context(
                formatted_question, sampled_caps, num_frames
            )

            inference_path = s2_parsed_response["thought"]
            answer = parsed_answer_s2(s2_parsed_response["answer"])
            # save_to_vector_db(self.fact_vector_db, sampled_caps)  # 存储相关事实和推测
            # save_to_vector_db(self.spec_vector_db, thought)

            # step4:反思评估器，校验推理过程是否合理，给出答案的置信度
            # previous_information = get_previous_information(inference_path)
            parsed_self_eval_response = self.self_eval(previous_prompt, inference_path, answer)
            # inference_path = parsed_self_eval_response["thought"]
            confidence = parse_self_eval_response_find_confidence(parsed_self_eval_response["answer"])
            if confidence == 3:
                score_flag = True
                final_answer = answer
                break
            i = i + 1
        # 返回答案
        if score_flag:
            return final_answer, count_frame
        else:
            logger.info(f"no answer in process video_id: {video_id}, guessed an answer")
            return random.randint(0, 4), count_frame


# ================= 使用示例 =================
if __name__ == "__main__":
    # 0.首次使用需要构建知识库
    # build_text_vector_db()

    # 1.初始化系统
    rag_system = MultiModalRAG()
