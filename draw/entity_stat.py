# """
# 不同实体数量下句子的分布, 不同种类的实体分布
# """
import os
from collections import Counter
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = u'SimHei'


def plot_entity_count(data: Counter, save_name, rotation=0, so=False):
    """
    将{'B-API': 9830, 'B-PL': 6040, 'B-Fram': 5900, 'B-Stan': 2910, 'B-Plat': 910}
    化成直方图
    :param save_name:
    :param data:
    :return:
    """
    data = data.most_common()
    keys = [x[0].split("-")[1] for x in data]
    for i in range(len(keys)):
        if keys[i] == "User_Interface_Element":
            keys[i] = "User_Interface"
    nums = [x[1] // 10 for x in data]
    # keys = [i for i in range(len(data))]
    plt.figure()
    fig, ax = plt.subplots()
    b = ax.bar(keys, nums)
    print("_______________________")
    for i in b:
        h = i.get_height()
        if rotation == 0 and not so:
            ax.text(i.get_x()+i.get_width()/4, h, "%d" % int(h), rotation=rotation)
        else:
            ax.text(i.get_x() - 0.5 + i.get_width() / 2, h + 50, "%d" % int(h), rotation=270)
    plt.xlabel('Entity type')
    plt.ylabel('Number of entities')
    plt.xticks(rotation=rotation)
    plt.show()
    save_path = os.getcwd()
    fig.savefig(os.path.join(save_path, save_name+".pdf"))


def get_ent_dist(file, entity_cnt):
    """
    统计一个文件中的实体数量
    :param entity_cnt:
    :param file:
    :return:
    """
    with open(file, 'r') as fr:
        for line in fr.readlines():
            line = line.strip().split()
            if not line:
                continue
            token, label = line[0], line[1]
            if "B" == label.split("-")[0]:
                entity_cnt[label] += 1
    print(entity_cnt)


def main():
    sner_dir = ["data/Annotated_training_testing_data/testset",
                "data/Annotated_training_testing_data/trainset"]
    sner_files = list()
    for d in sner_dir:
        temp = [os.path.join(d, x) for x in os.listdir(d)]
        sner_files.extend(temp)
    print(sner_files)
    entity_cnt = Counter()
    for f in sner_files:
        get_ent_dist(f, entity_cnt)
    plot_entity_count(entity_cnt, "sner_ent_dist")

    # so_files = ["data/annotated_ner_data/StackOverflow/dev.txt",
    #             "data/annotated_ner_data/StackOverflow/test.txt",
    #             "data/annotated_ner_data/StackOverflow/train.txt",
    #             # "data/annotated_ner_data/StackOverflow/train_merged_labels.txt"
    #             ]
    # entity_cnt = Counter()
    # for f in so_files:
    #     get_ent_dist(f, entity_cnt)
    # plot_entity_count(entity_cnt, "so_ent_dist_x", so=True)


if __name__ == '__main__':
    main()

