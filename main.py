import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor

from agent.langchain_workflow1 import MultiModalRAG
from agent.work_flow import run_one_question
from config import PROJECT_ROOT

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
    anns = json.load(open(input_q_a_file, "r"))
    all_caps = json.load(open(image_cap_file, "r"))

    task_id_list = list(anns.keys())  # 全量任务
    # step 初始化系统
    rag_system = MultiModalRAG()

    for i in range(500):
        video_id = task_id_list[i]

        # # 跳过已经得到答案的
        # output_json_path = os.path.join(
        #     PROJECT_ROOT,
        #     "data/output/answer",
        #     f"egoschema_subset_{video_id}.json"
        # )
        # if os.path.exists(output_json_path):
        #     continue

        # 回答问题
        run_one_question(video_id, anns[video_id], all_caps[video_id], rag_system)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
