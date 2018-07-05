import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn.utils.rnn import pack_padded_sequence

class Classifier(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, dropout=0.0):
        super(Classifier, self).__init__()
        self.drop1 = nn.Dropout(dropout)
        self.lin1 = nn.Linear(in_features, mid_features)
        self.relu = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)
        self.lin2 = nn.Linear(mid_features, out_features)

    def forward(self, x):
        x = self.drop1(x)
        x = self.lin1(x)
        x = self.relu(x)
        x = self.drop2(x)
        x = self.lin2(x)
        return x

class TextFeatures(nn.Module):
    def __init__(self, embedding_tokens, embedding_features, lstm_features, dropout=0.0):
        super(TextFeatures, self).__init__()
        self.embedding = nn.Embedding(embedding_tokens, embedding_features, padding_idx=0)
        self.drop = nn.Dropout(dropout)
        self.tanh = nn.Tanh()
        self.lstm = nn.LSTM(input_size=embedding_features,
                            hidden_size=lstm_features,
                            num_layers=1)
        self.features = lstm_features
        self._init(self.lstm, self.embedding)

    def _init(self, lstm, embedding):
        # lstm
        for w in lstm.weight_ih_l0.chunk(4,0):
            init.xavier_uniform_(w)
        lstm.bias_ih_l0.data.zero_()

        for w in lstm.weight_hh_l0.chunk(4,0):
            init.xavier_uniform_(w)
        lstm.bias_hh_l0.data.zero_()

        # embedding
        init.xavier_uniform_(embedding.weight)

    def forward(self, q, q_len):
        embedded = self.embedding(q)
        tanhed = self.tanh(self.drop(embedded))
        packed = pack_padded_sequence(tanhed, q_len, batch_first=True)
        _, (_, c) = self.lstm(packed)
        return c.squeeze(0)