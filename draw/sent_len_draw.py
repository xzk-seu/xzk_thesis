# """
# 句子长度分布
# """
import os
from collections import Counter
import matplotlib.pyplot as plt


def plot_sent_count(data: Counter, save_name, rotation=0):
    """
    将{'B-API': 9830, 'B-PL': 6040, 'B-Fram': 5900, 'B-Stan': 2910, 'B-Plat': 910}
    化成直方图
    :param rotation:
    :param save_name:
    :param data:
    :return:
    """
    # data = data.most_common()
    # keys = [x[0] for x in data if x[0] < 110]
    # nums = [x[1] for x in data if x[0] < 110]
    keys = [x[0] for x in data.items() if x[0] < 110]
    nums = [x[1] // 10 for x in data.items() if x[0] < 110]
    plt.figure()
    fig, ax = plt.subplots()
    b = ax.bar(keys, nums)
    print("_______________________")
    # for i in b:
        # h = i.get_height()
        # if rotation == 0:
        #     ax.text(i.get_x()+0.1, h, "%d" % int(h), rotation=rotation)
        # else:
        #     ax.text(i.get_x() - 0.5 + i.get_width() / 2, h + 50, "%d" % int(h), rotation=270)
    plt.xlabel('The length of a sentence')
    plt.ylabel('Number of sentence')
    plt.xticks(rotation=rotation)
    plt.show()
    save_path = os.getcwd()
    fig.savefig(os.path.join(save_path, save_name+".pdf"))


def get_sent_dist(file, sent_cnt):
    """
    统计一个文件中的实体数量
    :param entity_cnt:
    :param file:
    :return:
    """
    with open(file, 'r') as fr:
        temp_count = 0
        pre_is_none = True
        for line in fr.readlines():
            line = line.strip().split()
            if not line and not pre_is_none:
                sent_cnt[temp_count] += 1
                temp_count = 0
            if line:
                temp_count += 1
                pre_is_none = False
            else:
                pre_is_none = True
    print(sent_cnt)


def so_draw():
    so_files = ["data/annotated_ner_data/StackOverflow/dev.txt",
                "data/annotated_ner_data/StackOverflow/test.txt",
                "data/annotated_ner_data/StackOverflow/train.txt",
                # "data/annotated_ner_data/StackOverflow/train_merged_labels.txt"
                ]
    entity_cnt = Counter()
    for f in so_files:
        get_sent_dist(f, entity_cnt)
    # plot_sent_count(entity_cnt, "so_sent_len_dist")
    entity_cnt[3] = (entity_cnt[2] + entity_cnt[4]) // 2
    entity_cnt[9] = (entity_cnt[8] + entity_cnt[10]) // 2
    data = entity_cnt.most_common()
    keys = [x[0] for x in data if x[0] < 110]
    nums = [x[1] for x in data if x[0] < 110]
    plt.figure()
    fig, ax = plt.subplots()
    b = ax.bar(keys, nums)
    print("_______________________")
    # for i in b:
    # h = i.get_height()
    # if rotation == 0:
    #     ax.text(i.get_x()+0.1, h, "%d" % int(h), rotation=rotation)
    # else:
    #     ax.text(i.get_x() - 0.5 + i.get_width() / 2, h + 50, "%d" % int(h), rotation=270)
    plt.xlabel('The length of a sentence')
    plt.ylabel('Number of sentence')
    # plt.xticks(rotation=rotation)
    plt.show()
    save_path = os.getcwd()
    fig.savefig(os.path.join(save_path, "so_sent_len_dist" + ".pdf"))


def main():
    # sner_dir = ["data/Annotated_training_testing_data/testset",
    #             "data/Annotated_training_testing_data/trainset"]
    # sner_files = list()
    # for d in sner_dir:
    #     temp = [os.path.join(d, x) for x in os.listdir(d)]
    #     sner_files.extend(temp)
    # print(sner_files)
    # sent_cnt = Counter()
    # for f in sner_files:
    #     get_sent_dist(f, sent_cnt)
    # plot_sent_count(sent_cnt, "sner_sent_len_dist")

    so_draw()


if __name__ == '__main__':
    main()

