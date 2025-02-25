import logging
import time

from agent.langchain_workflow1_base import MultiModalRAG
from data.prompt.const_prompt import TEST_PROMPT

logger = logging.getLogger(__name__)


class MultiModalRAG_IRAG(MultiModalRAG):
    def ask_vlm(self, question, image_path, prompt=None):
        logger.info(f"start to ask_vlm, question: {question}")
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

def run_one_question(video_id, ann, caps, answer_json):
    """
    处理单个视频问题的函数。

    参数:
        video_id : 视频的唯一标识符
        ann : 问题
        caps : 视频每帧的字幕数据
        answer_json: 日志对象，用于记录最终答案。
    """
    logger.info("***********************************")
    logger.info(f"video_id ={video_id}, begin to run_one_question")
    # 初始化系统
    rag_system = MultiModalRAG()

    # 示例查询
    prompt = TEST_PROMPT
    response = rag_system.ask_vlm(
        question="why the image related to context, give me reasons?",
        image_path="./data/input/image/red_dream.png",
        prompt=None
    )
    logging.info(f"response: {response}")
    answer_json[video_id] = {
        "answer": 1,
        "label": 1,
        "corr": 1,
        "count_frame": 1,
    }
    logger.info(f"video_id ={video_id}, end to run_one_question")
    logger.info("***********************************")
