import json

if __name__ == '__main__':
    data = json.load(open("../data/output/egoschema_subset.json"))
    print(len(data))

    accs = []
    frames = []
    for key in data:
        acc = data[key]["answer"] == data[key]["label"]
        accs.append(acc)

        frame = data[key]["count_frame"]
        frames.append(frame)

    print("Mean accuracy: ", sum(accs) / len(accs))
    print("Mean frame: ", sum(frames) / len(frames))
