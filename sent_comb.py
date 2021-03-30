

f = "data/annotated_ner_data/StackOverflow/dev.txt"
vocab_set = set()
label_set = set()
with open(f, 'r') as fr:
    b = list()
    for line in fr.readlines(10000):
        t = line.split()
        if not t:
            print(b)
            b = list()
            continue
        b.append(t[0])
