import json
import matplotlib.pyplot as plt


def plot_count1(data):

    count = []
    label = []
    for k, v in data.items():
        count.append(k)
        label.append(v)
    sorted_info = sorted(enumerate(label), key=lambda n: -n[1])
    sorted_label = [i[1] for i in sorted_info]
    sorted_idx = [i[0] for i in sorted_info]
    sorted_count = [0] * len(label)
    for i, idx in enumerate(sorted_idx):
        sorted_count[i] = count[idx]
    plt.figure()
    fig, ax = plt.subplots()
    b = ax.bar(sorted_count[1:], sorted_label[1:])
    print("_______________________")
    for i in b:
        h = i.get_height()
        ax.text(i.get_x()+i.get_width()/4, h, "%d" % int(h))
    plt.xlabel('label')
    plt.ylabel('number')
    # plt.xticks(rotation=300)
    plt.title('Quantity of various labels')
    plt.show()


if __name__ == "__main__":
    with open("count/Annotated_training_testing_data/trainset.json", 'r') as fp:
        anno_train = json.load(fp)
    with open("count/Annotated_training_testing_data/testset.json", 'r') as fp:
        anno_test = json.load(fp)
    data = merge_dict(anno_train, anno_test)
    # 第一张图
    plot_count1(data)
