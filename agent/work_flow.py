import logging
import time
import json
import os

import cv2

from agent.langchain_workflow1 import clean_vector_db, parse_deepseek_response
from config import PROJECT_ROOT
from data.prompt.const_prompt import LLM_GET_ANSWER_SYSTEM_PROMPT, LLM_DEFAULT_SYSTEM_PROMPT

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


def test_rag_vlm(rag_system):
    # 1.示例rag查询VLM
    image_description = rag_system.ask_vlm(
        image_path=PROJECT_ROOT + "/data/input/image/red_dream.png",
        prompt="what is in the image?"
    )
    # 2.从向量数据库内召回最相关的文本
    context = rag_system.retrieve_from_vector_db(rag_system.fact_vector_db, image_description, k=3)
    # 3.根据召回，构建提示词，并回答相关问题
    final_prompt = LLM_GET_ANSWER_SYSTEM_PROMPT.format(
        context=context,
        question=""
    )
    rag_system.ask_vlm(
        image_path=PROJECT_ROOT + "/data/input/image/red_dream.png",
        prompt=final_prompt
    )


def test_llm(rag_system):
    question = "请写一个反转列表的代码"
    response = rag_system.ask_llm(prompt=question, system_prompt=LLM_DEFAULT_SYSTEM_PROMPT)
    parsed_response = parse_deepseek_response(response)
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


def save_json(data, file_path, ensure_ascii=False):
    """保存JSON数据到指定路径，自动创建不存在的目录

    Args:
        data: 要保存的Python对象
        file_path: 完整的文件保存路径
        ensure_ascii: 是否转义非ASCII字符（默认False）
    """
    # 创建父目录（如果不存在）
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # 使用with语句安全写入文件
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=ensure_ascii)


def run_one_question(video_id, ann, caps, rag_system):
    """
    处理单个视频问题的函数。

    参数:
        video_id : 视频的唯一标识符
        ann : 问题
        caps : 视频每帧的字幕数据
    """
    # 记录开始时间
    start_time = time.time()
    logger.info("***********************************")
    logger.info(f"video_id ={video_id},begin to run_one_question")
    question = ann["question"]
    options = [ann[f"option {i}"] for i in range(5)]
    # extract_frame(video_id) # 首次运行可以抽视频保存为图像，校验数据集的完整性、
    formatted_question = (
            f"Here is the question: {question}\n"
            + "Here are the choices: "
            + " ".join([f"{i}. {ans}" for i, ans in enumerate(options)])
    )
    logger.info(f"formatted_question = {formatted_question}")
    truth = ann["truth"]
    logger.info(f"truth ={truth}")

    # 运行agent，获取问题答案
    answer, count_frame = rag_system.get_answer(video_id, formatted_question, caps, truth)
    # 计算并打印执行时间
    end_time = time.time()
    execution_time = end_time - start_time

    # 保存并校验答案
    answer_json = {
        "answer": answer,
        "label": truth,
        "corr": str(answer) == str(truth),
        "count_frame": count_frame,
        "execution_time": execution_time,
    }
    output_json_path = os.path.join(
        PROJECT_ROOT,
        "data/output/answer",
        f"egoschema_subset_{video_id}.json"
    )
    save_json(answer_json, output_json_path)
    logger.info(f"video_id ={video_id}, end to run_one_question")
    logger.info("***********************************")
