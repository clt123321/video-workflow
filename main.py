import json
import logging
from concurrent.futures import ThreadPoolExecutor

from agent.work_flow import run_one_question

logger = logging.getLogger(__name__)

# 统一配置根日志记录器
logging.basicConfig(
    filename="./log/egoschema_subset.log",  # 目标文件路径
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s (line %(lineno)d)",
    datefmt='%Y-%m-%d %H:%M:%S'
)


def main():
    logger.info("Hello World")
    input_q_a_file = "data/input/subset_anno.json"  # 问题id和问题内容
    image_cap_file = "data/input/lavila_subset.json"  # 用lavila预处理，每帧的字幕文本
    output_json = "data/output/egoschema_subset.json"  # 运行结果存储

    anns = json.load(open(input_q_a_file, "r"))
    all_caps = json.load(open(image_cap_file, "r"))
    logs = {}

    # task_id_list = list(anns.keys())  # 运行全量任务
    task_id_list = [list(anns.keys())[0]]  # 运行单个任务
    tasks = [
        (video_id, anns[video_id], all_caps[video_id], logs)
        for video_id in task_id_list
    ]
    with ThreadPoolExecutor(max_workers=1) as executor:
        executor.map(lambda p: run_one_question(*p), tasks)

    json.dump(logs, open(output_json, "w"))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
