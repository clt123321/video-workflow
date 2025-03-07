import logging
import time
import os

import cv2

from config import PROJECT_ROOT
from data.prompt.const_prompt import DEFAULT_PROMPT

logger = logging.getLogger(__name__)


def process_video(video_path, target_frames=None, save_dir=None,
                  count_actual_frame=False):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError("can not open video, please check video path: " + video_path)

    # 获取视频元数据（某些视频的元数据可能不准确）
    total_frames_meta = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames_meta / fps

    # 实际逐帧统计（更准确）
    total_frames_actual = None
    if count_actual_frame:
        total_frames_actual = 0
        while True:
            ret, _ = cap.read()
            if not ret:
                break
            total_frames_actual += 1

    # 重置视频读取位置
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # 提取特定帧逻辑,默认使用均匀采样
    if target_frames is None:
        target_frames = generate_uniform_frames(fps, duration)
    if target_frames is not None:
        os.makedirs(save_dir, exist_ok=True)

        for frame_id in target_frames:
            if frame_id >= total_frames_meta:
                logger.info(f"warning: frame id: {frame_id} out of range")
                continue

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ret, frame = cap.read()
            if ret:
                output_path = f"{save_dir}/frame_{frame_id:05d}.jpg"
                cv2.imwrite(output_path, frame)
                logger.info(f"save frame {frame_id} into {output_path}")

    cap.release()

    return {
        "total_frames_meta": total_frames_meta,
        "total_frames_actual": total_frames_actual,
        "fps": fps,
        "duration_sec": duration
    }


def generate_uniform_frames(fps, duration):
    """
    生成均匀间隔的帧号列表
    :param fps: 帧率 (支持整数或浮点数)
    :param duration: 视频时长 (单位：秒)
    :return: 排序去重后的整数帧号列表
    """
    max_value = duration * fps  # 计算理论最大帧位置

    # 处理浮点精度问题 (避免 0.1 + 0.2 = 0.30000000000000004 等问题)
    current = 0.0
    frames = []

    # 使用十进制精度控制循环
    from decimal import Decimal, getcontext
    getcontext().prec = 10  # 设置足够高的精度

    step = Decimal(str(fps))
    current_dec = Decimal('0')
    max_dec = Decimal(str(max_value))

    while current_dec <= max_dec:
        # 四舍五入到最接近的整数
        frame_num = int(round(float(current_dec)))
        frames.append(frame_num)
        current_dec += step

    # 去重并排序
    return sorted(list(set(frames)))


def run_one_question(video_id, ann, caps, answer_json,rag_system):
    """
    处理单个视频问题的函数。

    参数:
        video_id : 视频的唯一标识符
        ann : 问题
        caps : 视频每帧的字幕数据
        answer_json: 日志对象，用于记录最终答案。
    """
    logger.info("***********************************")
    logger.info(f"video_id ={video_id},begin to run_one_question")
    question = ann["question"]
    options = [ann[f"option {i}"] for i in range(5)]
    # extract_frame(video_id)

    # rag_system.give_answer_method1(video_id, question, options)


    # 测试图像caption
    rag_system.ask_vlm(
        image_path=PROJECT_ROOT + "/data/input/image/red_dream.png",
        prompt="give me a caption about this image"
    )
    # 3.示例rag查询vlm
    # test_rag_vlm(rag_system)

    # 4.示例查询llm
    # test_llm(rag_system)

    # step3 保存并校验答案
    truth = ann["truth"]
    logger.info(f"truth ={truth}")
    answer_json[video_id] = {
        "answer": 1,
        "label": 1,
        "corr": 1,
        "count_frame": 1,
    }
    logger.info(f"video_id ={video_id}, end to run_one_question")
    logger.info("***********************************")


def test_rag_vlm(rag_system):
    # 1.示例rag查询VLM
    image_description = rag_system.ask_vlm(
        image_path=PROJECT_ROOT + "/data/input/image/red_dream.png",
        prompt="what is in the image?"
    )
    # 2.从向量数据库内召回最相关的文本
    context = rag_system.retrieve_from_vector_db(rag_system.vector_db, image_description, k=3)
    # 3.根据召回，构建提示词，并回答相关问题
    final_prompt = DEFAULT_PROMPT.format(
        context=context,
        question=""
    )
    rag_system.ask_vlm(
        image_path=PROJECT_ROOT + "/data/input/image/red_dream.png",
        prompt=final_prompt
    )


def test_llm(rag_system):
    question = "请写一个反转列表的代码"
    parsed_response = rag_system.ask_llm(question=question, system_prompt="you are a helpful ai assistant")
    logger.info("思考过程：\n" + parsed_response["thought"])
    logger.info("回答：\n" + parsed_response["answer"])

def extract_frame(video_id):
    stats = process_video(
        video_path="data/input/videos/sub_egoschema/" + video_id + ".mp4",
        save_dir="data/output/extracted_frames/" + video_id,
        count_actual_frame=True
    )
    logger.info(f"""
    video anaysis results：
    - total_frames_meta: {stats['total_frames_meta']}
    - total_frames_actual: {stats['total_frames_actual']}
    - FPS: {stats['fps']:.2f}
    - duration_sec: {stats['duration_sec']:.2f} seconds
    """)
