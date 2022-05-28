"""
将数据集中全O的进行匹配，看看能不能得到新的未标注的实体
"""


import os
from collections import Counter
import matplotlib.pyplot as plt
import json
import re


def get_new_entity_num(sent_label, so_dict):
    """
    能匹配上的新实体数量
    :return:
    """
    forbid = ["Question_ID", "with", "like", "this", "please", "This", "that", "your"]
    temp_count = 0
    for line in sent_label:
        word, label = line[0:2]
        if word in so_dict and label == "O" and word not in forbid:
            # print(word)
            temp_count += 1
    return temp_count


def get_sent_dist(file, sent_cnt, so_dict):
    """
    统计一个文件中的实体数量
    :return:
    """
    print(file)
    with open(file, 'r') as fr:
        temp_count = 0
        pre_is_none = True
        sent_label = list()
        entity_num = 0
        for line in fr.readlines():
            line = line.strip().split()
            if line:
                sent_label.append(line)
            if not line:
                if pre_is_none:
                    continue
                else:
                    # if temp_count > 10:
                    #     print(' '.join(sent))
                    if temp_count == 0:
                        entity_num += get_new_entity_num(sent_label, so_dict)
                    sent_label = list()
                    sent_cnt[temp_count] += 1
                    temp_count = 0
                pre_is_none = True
                continue
            pre_is_none = False
            token, label = line[0], line[1]
            if "B" == label.split("-")[0]:
                temp_count += 1
    print("从文件%s中\n发现%d个新实体" % (file, entity_num))
    print(sent_cnt)
    print(entity_num)


def main():

    so_files = ["data/annotated_ner_data/StackOverflow/dev.txt",
                "data/annotated_ner_data/StackOverflow/test.txt",
                "data/annotated_ner_data/StackOverflow/train.txt"
                ]
    so_dict = get_dict()
    for f in so_files:
        sent_cnt = Counter()
        get_sent_dist(f, sent_cnt, so_dict)


# def get_data_0(file):
#     """
#     获取一句话全O的句子
#     :return:
#     """
#     print(file)
#     with open(file, 'r') as fr:
#         temp_count = 0
#         pre_is_none = True
#         sent = list()
#         for line in fr.readlines():
#             line = line.strip().split()
#             if not line:
#                 continue


def get_dict():
    with open("draw/stackoverflow_dict.json", "r") as fr:
        data = json.load(fr)
        dict_set = set()
        for v in data.values():
            if len(v) > 0:
                for x in v:
                    dict_set.update(set(x))
        print(len(dict_set))
        dict_set = [x for x in dict_set if 30 > len(x) > 3]
        dict_set = set(dict_set)
        print(len(dict_set))

        dict_set = [x for x in dict_set if not bool(re.search(r'\d', x))]
        dict_set = set(dict_set)
        print(len(dict_set))

        word_len_count = Counter()
        for w in dict_set:
            word_len_count[len(w)] += 1
        return dict_set


if __name__ == '__main__':
    main()

