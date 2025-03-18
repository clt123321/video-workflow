import concurrent
import json
import logging
import queue
import threading
import time
from logging.handlers import QueueHandler, QueueListener
from concurrent.futures import ThreadPoolExecutor

from agent.langchain_workflow1 import MultiModalRAG
from agent.work_flow import run_one_question

# 初始化日志队列
log_queue = queue.Queue(-1)

# 配置统一格式的文件处理器
file_handler = logging.FileHandler(
    filename="./log/egoschema_subset.log",
    mode='a',
    encoding='utf-8'
)
formatter = logging.Formatter(
    fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s (line %(lineno)d)",
    datefmt='%Y-%m-%d %H:%M:%S'
)
file_handler.setFormatter(formatter)

# 创建队列监听器
queue_listener = QueueListener(log_queue, file_handler)
queue_listener.start()

# 配置根日志记录器（关键修改）
logger = logging.getLogger()
logger.addHandler(QueueHandler(log_queue))  # 所有日志通过队列传递
logger.setLevel(logging.INFO)  # 设置全局日志级别

# 取消传播限制（确保子模块日志能传递到根记录器）
logger.propagate = True  # 默认即为True，除非被其他代码修改

# 全局重试队列（线程安全）
retry_queue = queue.Queue()
# 创建后台重试线程
retry_thread = None


def start_retry_worker(executor):
    """后台重试工作线程"""
    while True:
        try:
            # 获取待重试任务（阻塞式）
            task = retry_queue.get(timeout=300)  # 5分钟无任务自动退出
            video_id, ann, caps, rag_system = task

            # 等待30秒后重新提交
            time.sleep(30)
            executor.submit(
                safe_process_video,
                video_id, ann, caps, rag_system
            )
            print(f"已重新提交视频处理任务 {video_id}")
        except queue.Empty:
            print("重试线程空闲超时，退出")
            break
        except Exception as e:
            print(f"重试线程异常: {str(e)}")


def safe_process_video(video_id, ann, caps, rag_system):
    """带异常隔离的任务包装函数"""
    try:
        process_video(video_id, ann, caps, rag_system)
    except ConnectionError as e:
        print(f"捕获到连接异常，将视频 {video_id} 加入重试队列")
        retry_queue.put((video_id, ann, caps, rag_system))


def process_video(video_id, ann, caps, rag_system):
    """带自定义日志记录的任务处理函数"""
    try:
        run_one_question(video_id, ann, caps, rag_system)
    except ConnectionError:
        # 直接抛出到包装函数处理
        print(f"捕获到连接异常，视频 {video_id} 抛出异常")
        raise
    except Exception as e:
        logger.error(f"Error in video {video_id}: {str(e)}", exc_info=True, stack_info=True)


def main():
    logger.info("start main")
    input_q_a_file = "data/input/subset_anno.json"  # 问题id和问题内容
    image_cap_file = "data/input/lavila_subset.json"  # 用lavila预处理，每帧的字幕文本
    anns = json.load(open(input_q_a_file, "r"))
    all_caps = json.load(open(image_cap_file, "r"))
    task_id_list = list(anns.keys())  # 全量任务
    # step 初始化系统
    rag_system = MultiModalRAG()
    start_time = time.time()

    global retry_thread

    # 创建线程池执行任务
    with ThreadPoolExecutor(max_workers=16) as executor:
        # 启动重试线程
        retry_thread = threading.Thread(
            target=start_retry_worker,
            args=(executor,),
            daemon=True
        )
        retry_thread.start()

        # 启动工作线程
        futures = [
            executor.submit(
                safe_process_video,
                video_id,
                anns[video_id],
                all_caps[video_id],
                rag_system
            )
            for video_id in task_id_list
        ]

        # 等待所有任务完成
        for future in concurrent.futures.as_completed(futures):
            future.result()  # 触发异常捕获

    # 确保所有日志写入完成
    queue_listener.stop()

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"total execution time: {execution_time}")

    # # 跳过已经得到答案的
    # output_json_path = os.path.join(
    #     PROJECT_ROOT,
    #     "data/output/answer",
    #     f"egoschema_subset_{video_id}.json"
    # )
    # if os.path.exists(output_json_path):
    #     continue


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
