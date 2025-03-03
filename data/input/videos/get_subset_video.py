import json

from tqdm import tqdm
import shutil
import os

def move_file(src, dst):
    try:
        # 检查源文件是否存在
        if not os.path.exists(src):
            print(f"源文件 {src} 不存在")
            return

        # 移动文件
        shutil.move(src, dst)
        print(f"文件已从 {src} 移动到 {dst}")
    except Exception as e:
        print(f"移动文件时出错: {e}")


if __name__ == "__main__":
    # Load necessary JSON files
    with open("../subset_anno.json") as questions_f:
        questions = json.load(questions_f)

    # Download videos
    for q_uid in tqdm(questions):
        print(q_uid)
        # 移动到当前文件夹下
        source_path = ".\egoschema-public\\videos\\videos"
        dest_path = "./sub_egoschema"
        move_file(source_path + "\\" + q_uid + ".mp4", dest_path)


