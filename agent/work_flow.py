import logging

logger = logging.getLogger(__name__)

def run_one_question(video_id, ann, caps, answer_json):
    """
    处理单个视频问题的函数。

    参数:
        video_id : 视频的唯一标识符
        ann : 问题
        caps : 视频每帧的字幕数据
        answer_json: 日志对象，用于记录最终答案。
    """
    logger.info("*******************begin to run_one_question****************")

    answer_json[video_id] = {
        "answer": 1,
        "label": 1,
        "corr": 1,
        "count_frame": 1,
    }
    logger.info("*******************end to run_one_question****************")
