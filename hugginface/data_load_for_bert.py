import json
from transformers import AutoTokenizer
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.utils.data.dataset import T_co

MAX_LEN = 512  # 句子的最大token长度
SENT_PADDING_VALUE = 0
LABEL_PADDING_VALUE = 50
bert_config_path = "BERTOverflow/config.json"
UNK = "[UNK]"
BERT_PATH = "./BERTOverflow"


class MyDataSet(torch.utils.data.Dataset):
    """
    生成结构为[[[w1, w2, w3], [l1, l2, l3]],[]]的数据集
    shape = (size, 2, sent_len)
    """

    def __init__(self, corpus_file_name, is_dense=True):
        """

        :param corpus_file_name: conll格式的文件
        :param is_dense: 是否过滤掉一句话中全是O的
        """
        super(MyDataSet).__init__()
        self.data = list()
        self.tokenizer = AutoTokenizer.from_pretrained(BERT_PATH)
        with open(bert_config_path, 'r') as fr:
            vocab = json.load(fr)
            l2idx = vocab["label2id"]
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

                mapping = {"B-Variable_Name": 'B-Variable',
                           "I-Variable_Name": 'I-Variable',
                           "B-Library_Class": "B-Library",
                           "I-Library_Class": "I-Library",
                           "B-Library_Function": "B-Function",
                           "I-Library_Function": "I-Function",
                           "B-Library_Variable": "B-Variable",
                           "I-Library_Variable": "I-Variable",
                           "B-Class_Name": "B-Class",
                           "I-Class_Name": "I-Class",
                           "B-Function_Name": "B-Function",
                           "I-Function_Name": "I-Function"}
                if t[1] in mapping:
                    t[1] = mapping[t[1]]
                label.append(l2idx[t[1]])
                x = self.tokenizer.convert_tokens_to_ids(t[0])
                sent.append(x)

        print(corpus_file_name)
        print("the size of dataset is ", len(sent_list))

        if not is_dense:
            return
        # 过滤掉一句话中全是O的
        o = l2idx['O']
        for sent, label in sent_list:
            for w in label:
                if w != o:
                    self.data.append([sent, label])
                    break

        print("过滤掉一句话中全是O的")
        print("the size of dataset is ", len(self.data))

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
    data_length = [len(xi[0]) for xi in batch_data]
    sent_seq = [torch.from_numpy(np.array(xi[0])) for xi in batch_data]
    label_seq = [torch.from_numpy(np.array(xi[1])) for xi in batch_data]
    padded_sent_seq = pad_sequence(sent_seq, batch_first=True, padding_value=SENT_PADDING_VALUE)
    padded_label_seq = pad_sequence(label_seq, batch_first=True, padding_value=LABEL_PADDING_VALUE)
    masks = torch.zeros(padded_sent_seq.shape, dtype=torch.uint8)
    for e_id, src_len in enumerate(data_length):
        masks[e_id, :src_len] = 1
    return padded_sent_seq, padded_label_seq, masks


class MyDataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        super().__init__(dataset, batch_size=batch_size, collate_fn=collate_fn)


if __name__ == '__main__':
    corpus_file = "data/annotated_ner_data/StackOverflow/dev.txt"
    ds = MyDataSet(corpus_file_name=corpus_file)
    dataloader = DataLoader(ds, batch_size=8, collate_fn=collate_fn)
    for idx, (sent, label, length) in enumerate(dataloader):
        print(idx)
        print(sent)
        print(label)
        print(length)
        break
    ds = MyDataSet(corpus_file_name="data/annotated_ner_data/StackOverflow/train.txt")
    tds = MyDataSet(corpus_file_name="data/annotated_ner_data/StackOverflow/test.txt")
