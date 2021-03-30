import json

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.utils.data.dataset import T_co

MAX_LEN = 512  # 句子的最大token长度
PADDING_VALUE = 0


class MyDataSet(torch.utils.data.Dataset):
    """
    生成结构为[[[w1, w2, w3], [l1, l2, l3]],[]]的数据集
    shape = (size, 2, sent_len)
    """

    def __init__(self, corpus_file_name, vocab_file_name):
        super(MyDataSet).__init__()
        self.data = list()
        with open(vocab_file_name, 'r') as fr:
            vocab = json.load(fr)
            v2idx = vocab["v2idx"]
            l2idx = vocab["l2idx"]
        with open(corpus_file_name, 'r') as fr:
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
                label.append(l2idx[t[1]])
                if t[0] not in v2idx:
                    sent.append(v2idx["<UNK>"])
                else:
                    sent.append(v2idx[t[0]])

        # 过滤掉一句话中全是O的
        o = l2idx['O']
        for sent, label in sent_list:
            for w in label:
                if w != o:
                    self.data.append([sent, label])
                    break

        print(corpus_file_name)
        print("this dataset is ", len(self.data))

    def __getitem__(self, index) -> T_co:
        return self.data[index]

    def __len__(self):
        return len(self.data)


# "sent_label_idx.json"
#
# train_dataloader = DataLoader(dataset, batch_size=8)
# test_dataloader = DataLoader(dataset[29308:], batch_size=8)

def collate_fn(batch_data):
    """
    自定义 batch 内各个数据条目的组织方式
    :param batch_data: [[[w1, w2, w3], [l1, l2, l3]],[[], []]]
    :return: 填充后的句子列表、实际长度的列表、以及label列表
    """
    batch_data.sort(key=lambda xi: len(xi[0]), reverse=True)
    sent_seq = [torch.from_numpy(np.array(xi[0])) for xi in batch_data]
    label_seq = [torch.from_numpy(np.array(xi[1])) for xi in batch_data]
    padded_sent_seq = pad_sequence(sent_seq, batch_first=True, padding_value=0)
    padded_label_seq = pad_sequence(label_seq, batch_first=True, padding_value=0)
    # t = padded_sent_seq.numpy().tolist()
    # t1 = padded_label_seq.numpy().tolist()
    return padded_sent_seq, padded_label_seq


class MyDataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        super().__init__(dataset, batch_size=batch_size, collate_fn=collate_fn)


if __name__ == '__main__':
    corpus_file = "data/annotated_ner_data/StackOverflow/dev.txt"
    v_file = "vocab.json"
    ds = MyDataSet(corpus_file_name=corpus_file, vocab_file_name=v_file)
    dataloader = DataLoader(ds, batch_size=8, collate_fn=collate_fn)
    for idx, (sent, label) in enumerate(dataloader):
        print(idx)
        print(sent)
        print(label)
        break
    ds = MyDataSet(corpus_file_name="data/annotated_ner_data/StackOverflow/test.txt", vocab_file_name=v_file)
