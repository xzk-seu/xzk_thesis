

# markdown标签统计
label_set = set()
with open('data/annotated_ner_data/StackOverflow/dev.txt', 'r') as fr:
    for line in fr.readlines():
        t = line.split()
        if not t:
            continue
        t = t[3]
        if t != 'O':
            label_set.add(t)
print(label_set)
print(len(label_set))

# NER标签统计
label_set = set()
with open('data/annotated_ner_data/StackOverflow/dev.txt', 'r') as fr:
    for line in fr.readlines():
        t = line.split()
        if not t:
            continue
        t = t[1]
        if t != 'O':
            label_set.add(t)
print(label_set)
print(len(label_set))

r = [x.split('-')[1] for x in label_set]
r = set(r)
print(r)
print(len(r))

s = [x for x in r if 'I-'+x not in label_set]
print(s)

