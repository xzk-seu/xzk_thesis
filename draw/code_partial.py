"""
统计代码块实体词汇，占总词汇的总数

"""

from collections import Counter


def read_file(file):
    print(file)
    code_num = 0
    token_num = 0
    o_num = 0
    with open(file, 'r') as fr:

        for line in fr.readlines():
            line = line.strip().split()
            if not line:
                continue
            word, label = line[0:2]
            token_num += 1
            if "Code_Block" in label:
                code_num += 1
            if label == "O":
                o_num += 1
    return code_num, token_num, o_num


def main():

    so_files = ["data/annotated_ner_data/StackOverflow/dev.txt",
                "data/annotated_ner_data/StackOverflow/test.txt",
                "data/annotated_ner_data/StackOverflow/train.txt"
                ]
    code_num, token_num = 0, 0
    o_num = 0
    for f in so_files:
        a, b, c = read_file(f)
        code_num += a
        token_num += b
        o_num += c
    print("code_num: ", code_num)
    print("token_num: ", token_num)
    print("o_num: ", o_num)
    print(o_num / token_num)
    print(code_num / token_num)


if __name__ == '__main__':
    main()
