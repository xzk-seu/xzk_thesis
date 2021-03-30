import json

import torch
from seqeval.metrics import accuracy_score
from seqeval.metrics import f1_score
from seqeval.metrics import precision_score
from seqeval.metrics import recall_score
from torch import nn
from torchcrf import CRF
from tqdm import tqdm

from data_load import MyDataSet, MyDataLoader

BATCH_SIZE = 8
NUM_CLASS = 55  # 标签数量
EMB_DIM = 64  # 嵌入维度
EMB_TABLE_SIZE = 7000
MAX_LEN = 512  # 句子的最大token长度

device = "cuda" if torch.cuda.is_available() else "cpu"


# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.emb = nn.Embedding(EMB_TABLE_SIZE, embedding_dim=EMB_DIM)
        self.rnn = nn.GRU(input_size=EMB_DIM, hidden_size=512, num_layers=2, batch_first=True, bidirectional=True)
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(512 * 2, NUM_CLASS),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.emb(x)
        h0 = torch.randn(2 * 2, x.size(0), 512)
        h0 = h0.to(device)
        output, hn = self.rnn(x, h0)
        output = self.linear_relu_stack(output)
        return output


model = NeuralNetwork().to(device)
crf = CRF(NUM_CLASS, batch_first=True).to(device)
print(model)

# %%

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
with open('vocab.json', 'r') as fr:
    label_list = json.load(fr)["label_list"]


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    model.zero_grad()
    for batch, (X, y) in tqdm(enumerate(dataloader)):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        output = model(X)
        log_likelihood = crf(output, y)
        z = torch.zeros(log_likelihood.size())
        loss = z - log_likelihood

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model):
    size = len(dataloader.dataset)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        all_pred_y = list()
        all_y = list()
        print("testing")
        for X, y in tqdm(dataloader):
            X, y = X.to(device), y.to(device)
            output = model(X)
            loss = 0 - crf(output, y)
            test_loss += loss
            pred_y = crf.decode(output)
            all_pred_y.extend(pred_y)
            all_y.extend(y.cpu().numpy().tolist())
    all_pred_y_label = [[label_list[t1] for t1 in t2] for t2 in all_pred_y]
    all_y_label = [[label_list[t1] for t1 in t2] for t2 in all_y]
    print('p', precision_score(all_pred_y_label, all_y_label))
    print('r', recall_score(all_pred_y_label, all_y_label))
    print('f1', f1_score(all_pred_y_label, all_y_label))
    print('acc', accuracy_score(all_pred_y_label, all_y_label))
    test_loss /= size
    print(f"Avg loss: {test_loss:>8f} \n")


epochs = 5
corpus_file = "data/annotated_ner_data/StackOverflow/dev.txt"
test_file = "data/annotated_ner_data/StackOverflow/test.txt"
vocab_file_name = "vocab.json"
train_set = MyDataSet(corpus_file_name=corpus_file, vocab_file_name=vocab_file_name)
test_set = MyDataSet(corpus_file_name=test_file, vocab_file_name=vocab_file_name)
train_dataloader = MyDataLoader(train_set, batch_size=BATCH_SIZE)
test_dataloader = MyDataLoader(test_set, batch_size=BATCH_SIZE)
for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model)
print("Done!")
