import json


def test_json_length():
    input_q_a_file = "../data/input/subset_anno.json"  # 问题id和问题内容
    image_cap_file = "../data/input/lavila_subset.json"  # 用lavila预处理，每帧的字幕文本
    anns = json.load(open(input_q_a_file, "r"))
    all_caps = json.load(open(image_cap_file, "r"))
    print(f"问题的数量：{len(anns)}" )
    print(f"视频每帧文本的数量：{len(all_caps)}" )

if __name__ == '__main__':
    test_json_length()
