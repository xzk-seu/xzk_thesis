"""
读取训练集语料库，生成vocab.json
包括词汇及标签的列表vocab_list，label_list
及其相应的倒排字典v2idx，l2idx

vocab_size = 6620
label_size = 55

"""

import json

f = "data/annotated_ner_data/StackOverflow/dev.txt"
vocab_set = set()
label_set = set()
with open(f, 'r') as fr:
    for line in fr.readlines():
        t = line.split()
        if not t:
            continue
        vocab_set.add(t[0])
        label_set.add(t[1])

label_set.add("I-HTML_XML_Tag")
label_set.add("I-File_Type")

print(len(vocab_set), len(label_set))
vocab_list = ["<PAD>", "<UNK>"]
vocab_list.extend(vocab_set)
label_list = ["O"]
label_set.remove("O")
label_list.extend(label_set)
v2idx = {vocab_list[i]: i for i in range(len(vocab_list))}
l2idx = {label_list[i]: i for i in range(len(label_list))}
# print(v2idx)
# print(l2idx)
print(len(vocab_list), len(label_list))
with open("vocab.json", 'w') as fw:
    json.dump({"label_list": label_list, "vocab_list": vocab_list, "v2idx": v2idx, "l2idx": l2idx}, fw)

res = dict()
with open(f, 'r') as fr:
    sent = list()
    label = list()
    sent_list = list()
    for line in fr.readlines():
        t = line.split()
        if not t:
            if sent:
                sent_list.append([sent, label])
                sent = list()
                label = list()
            continue
        sent.append(v2idx[t[0]])
        label.append(l2idx[t[1]])
print(len(sent_list))


# 过滤掉一句话中全是O的
o = l2idx['O']
sent_list_new = list()
for sent, label in sent_list:
    for w in label:
        if w != o:
            sent_list_new.append([sent, label])
            break
print(len(sent_list_new))

with open("sent_label_idx.json", 'w') as fw:
    json.dump(sent_list_new, fw)
