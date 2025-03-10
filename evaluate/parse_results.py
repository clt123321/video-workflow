import json
import os
from glob import glob

from config import PROJECT_ROOT


def calculate_corr_true_ratio(folder):
    # 获取所有JSON文件（仅单层目录）
    files = glob(os.path.join(folder, "*.json"))
    if not files:
        return None

    cnt = 0
    for f in files:
        with open(f, 'r', encoding='utf-8') as file:
            # 直接检查顶层是否存在 "corr": false
            if json.load(file).get('corr') is True:
                cnt += 1
    print(f"cnt:{cnt}, files_num:{len(files)}")
    return cnt / len(files) if cnt else None


if __name__ == '__main__':
    output_json_path = os.path.join(
        PROJECT_ROOT,
        "data/output/answer"
    )
    accuracy = calculate_corr_true_ratio(output_json_path)
    print(f"Mean accuracy: {accuracy}")
