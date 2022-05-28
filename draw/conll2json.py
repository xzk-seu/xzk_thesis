import json

"""
统计每个文件下各类实体token的数量

"""

from collections import Counter


def read_file(file):
    print(file)
    with open(file, 'r') as fr:
        pre_is_none = True
        sent = list()
        res = list()
        for line in fr.readlines():
            line = line.strip().split()
            if line:
                sent.append(line[0:2])
                pre_is_none = False
            if not line:
                if pre_is_none:
                    continue
                else:
                    res.append(sent)
                    sent = list()
                pre_is_none = True
    return res


def pick_sent(data: list):
    for sent in data:
        ent_count = 0
        for w, l in sent:
            if l.split("-")[0] == "I" and "Block" not in l:
                ent_count += 1
        if ent_count >= 3 and len(sent) < 200:
            print(sent)


def get_vocab():
    res = set()
    with open("BERTOverflow/vocab.txt", "r") as fr:
        for line in fr.readlines():
            res.add(line.strip())
    return res


def pick_unseen_word(data: list, vocab: set):
    for sent in data:
        ent_count = 0
        for w, l in sent:
            if "Block" not in l and l != "O" and w == "ToList()":
                print(sent)
                print(" ".join([x[0] for x in sent]))


def main():
    so_files = ["data/annotated_ner_data/StackOverflow/dev.txt",
                "data/annotated_ner_data/StackOverflow/test.txt",
                "data/annotated_ner_data/StackOverflow/train.txt"
                ]
    # t = read_file(so_files[0])
    # # t = read_file("data/Annotated_training_testing_data/testset/test3.conll")
    # # print(len(t))
    # with open("test.json", "w") as fw:
    #     json.dump(t, fw)
    # pick_sent(t)
    v = get_vocab()
    for f in so_files:
        t = read_file(f)
        pick_unseen_word(t, v)
        # pick_sent(t)
    # code_num, token_num = 0, 0
    # o_num = 0
    # for f in so_files:
    #     a, b, c = read_file(f)
    #     code_num += a
    #     token_num += b
    #     o_num += c
    # print("code_num: ", code_num)
    # print("token_num: ", token_num)
    # print("o_num: ", o_num)
    # print(o_num / token_num)
    # print(code_num / token_num)


if __name__ == '__main__':
    main()
