import json
import os
import re
from glob import glob

from config import PROJECT_ROOT


def calculate_corr_true_ratio(folder):
    # 获取所有JSON文件（仅单层目录）
    files = glob(os.path.join(folder, "*.json"))
    if not files:
        return None

    cnt = 0
    time = 0
    frame = 0
    for f in files:
        with open(f, 'r', encoding='utf-8') as file:
            data = json.load(file)  # 解析整个 JSON 文件为字典
            execution_time = data.get("execution_time")
            time += execution_time
            count_frame = data.get("count_frame")
            frame += count_frame
            if data.get('corr') is True:
                cnt += 1
    print(f"true answer:{cnt}, question_num:{len(files)}")
    print(f"time_total:{time}, average time:{time / len(files)}")
    print(f"frame_total:{frame}, average frame:{frame / len(files)}")
    return cnt / len(files) if cnt else None


def extract_video_ids(log_path):
    video_ids = []
    pattern = re.compile(r'video_id: ([a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12})')

    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            if "INFO - no answer in process video_id:" in line:
                match = pattern.search(line)
                if match:
                    video_ids.append(match.group(1))

    return video_ids


def count_string_occurrences(filename=PROJECT_ROOT + "/log/egoschema_subset.log", target=None):
    count = 0
    with open(filename, 'r', encoding='utf-8') as f:  # 指定编码为 utf-8
        for line in f:
            count += line.count(target)
    return count


if __name__ == '__main__':
    output_json_path = os.path.join(
        PROJECT_ROOT,
        "data/output/answer"
    )
    accuracy = calculate_corr_true_ratio(output_json_path)
    print(f"Mean accuracy: {accuracy}")

    TP = count_string_occurrences(target="correct_answer and confidence == 3")
    FN = count_string_occurrences(target="correct_answer and confidence != 3")
    FP = count_string_occurrences(target="correct_answer is false and confidence == 3")
    TN = count_string_occurrences(target="correct_answer is false is false and confidence != 3")
    ALL = TP + FP + FN + TN
    print(f"TP: {TP},FN: {FN},FP: {FP},TN: {TN},ALL: {ALL}")
    if ALL != 500:
        exit()
    ac = (TP + TN) / ALL
    precise_rate = TP / (TP + FP)
    recall_rate = TP / (TP + FN)
    Specificity = TN / (TN + FP)
    print(f"Precision: {precise_rate:.2f}")
    print(f"precise_rate: {precise_rate:.2f}")
    print(f"recall_rate: {recall_rate:.2f}")
    print(f"Specificity: {Specificity:.2f}")
