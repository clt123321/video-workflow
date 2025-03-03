import json
import logging
from concurrent.futures import ThreadPoolExecutor

from agent.work_flow import run_one_question

logger = logging.getLogger(__name__)

# 手动创建带UTF-8编码的文件处理器
file_handler = logging.FileHandler(
    filename="./log/egoschema_subset.log",
    mode='a',  # 追加模式
    encoding='utf-8'  # 强制指定编码
)

# 配置格式器
formatter = logging.Formatter(
    fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s (line %(lineno)d)",
    datefmt='%Y-%m-%d %H:%M:%S'
)
file_handler.setFormatter(formatter)

# 配置根日志记录器
root_logger = logging.getLogger()
root_logger.addHandler(file_handler)
root_logger.setLevel(logging.INFO)


def main():
    logger.info("start main")
    input_q_a_file = "data/input/subset_anno.json"  # 问题id和问题内容
    image_cap_file = "data/input/lavila_subset.json"  # 用lavila预处理，每帧的字幕文本
    output_json = "data/output/egoschema_subset.json"  # 运行结果存储

    anns = json.load(open(input_q_a_file, "r"))
    all_caps = json.load(open(image_cap_file, "r"))
    answer_json = {}

    task_id_list = list(anns.keys())  # 全量任务

    for i in range(1):
        video_id = task_id_list[i]
        run_one_question(video_id, anns[video_id], all_caps[video_id], answer_json)
        json.dump(answer_json, open(output_json, "w"))

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
