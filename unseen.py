import os


def get_word_set(conll_file):
    res = set()
    with open(conll_file, 'r') as fr:
        for line in fr.readlines():
            t = line.split()
            if not t:
                continue
            res.add(t[0])
    return res


def run(vocab_file, data_set_files):
    word_set = set()
    vocab = set()
    with open(vocab_file, 'r') as fr:
        for line in fr.readlines():
            vocab.add(line.strip())
    for f in data_set_files:
        word_set.update(get_word_set(f))
        print(len(word_set))

    res = [x for x in word_set if x.lower() not in vocab]
    print(len(res))


if __name__ == '__main__':
    # v_f = "BERTOverflow/vocab.txt"
    v_f = "vocab/bert_base_uncase_vocab.txt"

    ds_path = "data/annotated_ner_data/StackOverflow"
    ds_fs = os.listdir(ds_path)
    ds_fs = [os.path.join(ds_path, x) for x in ds_fs]
    print(ds_fs)
    run(v_f, ds_fs)
