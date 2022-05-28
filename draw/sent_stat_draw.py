# """
# 不同实体数量下句子的分布, 不同种类的实体分布
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
    # keys = [x[0] for x in data]
    # nums = [x[1] for x in data]
    # for k, v in data.items():
    keys = list(range(10))
    nums = [data.setdefault(i, 0)for i in keys]

    plt.figure()
    fig, ax = plt.subplots()
    b = ax.bar(keys, nums)
    print("_______________________")
    # for i in b:
    #     h = i.get_height()
    #     if rotation == 0:
    #         ax.text(i.get_x()-0.2, h, "%d" % int(h), rotation=rotation)
    #     else:
    #         ax.text(i.get_x() - 0.5 + i.get_width() / 2, h + 50, "%d" % int(h), rotation=270)
    plt.xlabel('The number of entities contained in a sentence')
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
    print(file)
    with open(file, 'r') as fr:
        temp_count = 0
        pre_is_none = True
        sent = list()
        for line in fr.readlines():
            line = line.strip().split()
            if line:
                sent.append(line[0])
            if not line:
                if pre_is_none:
                    continue
                else:
                    # if temp_count > 10:
                    #     print(' '.join(sent))
                    # sent = list()
                    sent_cnt[temp_count] += 1
                    temp_count = 0
                pre_is_none = True
                continue
            pre_is_none = False
            token, label = line[0], line[1]
            if "B" == label.split("-")[0]:
                temp_count += 1
    print(sent_cnt)


def main():
    # sner_dir = ["data/Annotated_training_testing_data/testset",
    #             "data/Annotated_training_testing_data/trainset"]
    # sner_files = list()
    # for d in sner_dir:
    #     temp = [os.path.join(d, x) for x in os.listdir(d)]
    # #     sner_files.extend(temp)
    # sner_files = ["data/Annotated_training_testing_data/testset/test0.conll",
    #               "data/Annotated_training_testing_data/trainset/train0.conll"]
    #
    # print(sner_files)
    # sent_cnt = Counter()
    # for f in sner_files:
    #     get_sent_dist(f, sent_cnt)
    # plot_sent_count(sent_cnt, "sner_sent_dist")

    so_files = ["data/annotated_ner_data/StackOverflow/dev.txt",
                "data/annotated_ner_data/StackOverflow/test.txt",
                "data/annotated_ner_data/StackOverflow/train.txt",
                # "data/annotated_ner_data/StackOverflow/train_merged_labels.txt"
                ]
    # entity_cnt = Counter()
    for f in so_files:
        entity_cnt = Counter()
        get_sent_dist(f, entity_cnt)
        s = 0
        for k, v in entity_cnt.items():
            s += k * v
        print(s)
    # plot_sent_count(entity_cnt, "so_sent_dist")


if __name__ == '__main__':
    main()

