import json

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from data_load import MyDataSet, MyDataLoader

torch.manual_seed(1)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("use", device)
BATCH_SIZE = 4
START_TAG = "<START>"
STOP_TAG = "<STOP>"
EMBEDDING_DIM = 32
HIDDEN_DIM = 64

LSTM_LAYERS = 1


def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, -1)
    return idx


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    # max_score = vec[:, 0, argmax(vec)]
    max_score = [v[0, argmax(v)] for v in vec]
    max_score = torch.Tensor(max_score)
    max_score = max_score.to(device)
    max_score_broadcast = max_score.view(BATCH_SIZE, 1, -1).expand(BATCH_SIZE, 1, vec.size()[-1])
    return max_score + \
           torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


class BiLSTM_CRF(nn.Module):

    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=LSTM_LAYERS, bidirectional=True, batch_first=True)

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

        self.hidden = self.init_hidden(BATCH_SIZE)

    def init_hidden(self, n):
        return (torch.randn(LSTM_LAYERS * 2, n, self.hidden_dim // 2).to(device),
                torch.randn(LSTM_LAYERS * 2, n, self.hidden_dim // 2).to(device))

    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((1, self.tagset_size), -10000.)
        # START_TAG has all of the score.
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        init_alphas = init_alphas.to(device)
        forward_var = init_alphas

        # Iterate through the sentence
        for feat in feats.permute([1, 0, 2]):
            alphas_t = []  # The forward tensors at this timestep
            for next_tag in range(self.tagset_size):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                # emit_score = feat[:, next_tag].view(
                #     1, -1).expand(1, self.tagset_size)
                emit_score = feat[:, next_tag]
                emit_score = emit_score.view(BATCH_SIZE, 1, -1).expand(BATCH_SIZE, 1, self.tagset_size)
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1)
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                t = log_sum_exp(next_tag_var)
                t = t.view(-1, 1)
                alphas_t.append(t)
            forward_var = torch.cat(alphas_t).view(BATCH_SIZE, 1, -1)
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden(sentence.size(0))
        # embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        embeds = self.word_embeds(sentence)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        # lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = torch.zeros(BATCH_SIZE, 1).to(device)
        temp_tensor = torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long).to(device)
        temp_tensor = temp_tensor.expand(BATCH_SIZE, 1)
        tags = torch.cat([temp_tensor, tags], dim=-1)
        for i, feat in enumerate(feats.permute([1, 0, 2])):
            for b in range(BATCH_SIZE):
                tran = self.transitions[tags[b, i + 1], tags[b, i]]
                score[b] = score[b] + tran + feat[b, tags[b, i + 1]]
        for b in range(BATCH_SIZE):
            score[b] = score[b] + self.transitions[self.tag_to_ix[STOP_TAG], tags[b, -1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []
        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tagset_size), -10000.)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        gold_score = gold_score.flatten()
        return forward_score - gold_score

    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence)

        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq


with open('vocab.json', 'r') as fr:
    vocab = json.load(fr)
    word_to_ix = vocab['v2idx']
    tag_to_ix = vocab['l2idx']
    tag_to_ix[START_TAG] = len(tag_to_ix)
    tag_to_ix[STOP_TAG] = len(tag_to_ix)

model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)
model.to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)


# Check predictions before training
# with torch.no_grad(training_data):
#     precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
#     precheck_tags = torch.tensor([tag_to_ix[t] for t in training_data[0][1]], dtype=torch.long)
#     print(model(precheck_sent))


def train(training_data):
    for batch, (sentence, tags) in tqdm(enumerate(training_data)):
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.train()
        model.zero_grad()
        sentence, tags = sentence.to(device), tags.to(device)

        # Step 2. Get our inputs ready for the network, that is,
        # turn them into Tensors of word indices.
        # sentence_in = prepare_sequence(sentence, word_to_ix)
        # targets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long)

        # Step 3. Run our forward pass.
        # loss = model.neg_log_likelihood(sentence_in, targets)
        loss = model.neg_log_likelihood(sentence, tags).sum() / BATCH_SIZE

        # Step 4. Compute the loss, gradients, and update the parameters by
        # calling optimizer.step()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(sentence)
            print(f"loss: {loss:>7f}  [{current:>5d}/{len(training_data.dataset):>5d}]")


corpus_file = "data/annotated_ner_data/StackOverflow/dev.txt"
test_file = "data/annotated_ner_data/StackOverflow/test.txt"
vocab_file_name = "vocab.json"
train_set = MyDataSet(corpus_file_name=corpus_file, vocab_file_name=vocab_file_name)
test_set = MyDataSet(corpus_file_name=test_file, vocab_file_name=vocab_file_name)
train_dataloader = MyDataLoader(train_set, batch_size=BATCH_SIZE)
test_dataloader = MyDataLoader(test_set, batch_size=BATCH_SIZE)
# Make sure prepare_sequence from earlier in the LSTM section is loaded
for epoch in range(5):  # again, normally you would NOT do 300 epochs, it is toy data
    print(f"Epoch {epoch + 1}\n-------------------------------")
    train(train_dataloader)
    # test(train_dataloader, model)
