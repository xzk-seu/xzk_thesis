import json

import torch
from seqeval.metrics import accuracy_score
from seqeval.metrics import f1_score
from seqeval.metrics import precision_score
from seqeval.metrics import recall_score
from torch import nn
from torchcrf import CRF
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from hugginface.data_load_for_bert import MyDataSet, MyDataLoader

BATCH_SIZE = 4
NUM_CLASS = 55  # 标签数量
BERT_PATH = "./BERTOverflow"
BERT_HIDDEN_SIZE = 768
HIDDEN_SIZE = 512
LEARNING_RATE = 1e-3
EPOCHS = 5

device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"
print("Use device", device)


# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.bert_model = AutoModel.from_pretrained(BERT_PATH)
        self.tokenizer = AutoTokenizer.from_pretrained(BERT_PATH)
        self.rnn = nn.GRU(input_size=BERT_HIDDEN_SIZE, hidden_size=HIDDEN_SIZE, num_layers=2,
                          batch_first=True, bidirectional=True)
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(HIDDEN_SIZE * 2, NUM_CLASS),
            nn.ReLU()
        )
        self.crf = CRF(NUM_CLASS, batch_first=True)

    def forward(self, x, y, masks):
        bert_embedding = self.bert_model(x).last_hidden_state
        h0 = torch.randn(2 * 2, x.size(0), HIDDEN_SIZE)
        h0 = h0.to(device)
        output, hn = self.rnn(bert_embedding, h0)
        output = self.linear_relu_stack(output)
        log_likelihood = self.crf(output, y, mask=masks)
        z = torch.zeros(log_likelihood.size())
        loss = z - log_likelihood
        pred_y = self.crf.decode(output, mask=masks)
        return loss, pred_y


model = NeuralNetwork().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
print(model)


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    model.zero_grad()
    for batch, (X, y, masks) in tqdm(enumerate(dataloader)):
        X, y = X.to(device), y.to(device)
        masks = masks.to(device)
        # Compute prediction error
        # output = model(X)
        # log_likelihood = crf(output, y)
        # z = torch.zeros(log_likelihood.size())
        # loss = z - log_likelihood
        loss, _ = model(X, y, masks)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, label_list):
    size = len(dataloader.dataset)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        all_pred_y = list()
        all_y = list()
        print("testing")
        for X, y, masks in tqdm(dataloader):
            X, y = X.to(device), y.to(device)
            masks = masks.to(device)
            # output = model(X)
            # loss = 0 - crf(output, y)
            loss, pred_y = model(X, y, masks)
            test_loss += loss
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


def main():
    train_file = "data/annotated_ner_data/StackOverflow/dev.txt"
    test_file = "data/annotated_ner_data/StackOverflow/test.txt"
    with open("BERTOverflow/config.json", 'r') as fr:
        id2label = json.load(fr)['id2label']
        label_list = [None for _ in range(NUM_CLASS)]
        for i in range(NUM_CLASS):
            label_list[i] = id2label[str(i)]
    train_set = MyDataSet(corpus_file_name=train_file)
    test_set = MyDataSet(corpus_file_name=test_file)
    train_dataloader = MyDataLoader(train_set, batch_size=BATCH_SIZE)
    test_dataloader = MyDataLoader(test_set, batch_size=BATCH_SIZE)
    for t in range(EPOCHS):
        print(f"Epoch {t + 1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model, label_list)
    print("Done!")


if __name__ == '__main__':
    main()
