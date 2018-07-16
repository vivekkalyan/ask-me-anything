import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

import config


class MainModel(nn.Module):
    def __init__(self, embedding_tokens):
        super(MainModel, self).__init__()

        question_features = 1024
        glimpses = 2
        image_features = config.output_features

        self.text = TextFeatures(
            embedding_tokens=embedding_tokens,
            embedding_features=300,
            lstm_features=question_features,
            dropout=0.5)

        self.attention = Attention(
            image_features=image_features,
            question_features=question_features,
            mid_features=512,
            glimpses=glimpses,
            dropout=0.5)

        self.avgpool = nn.AvgPool2d(
            kernel_size=config.output_size)

        self.classifier = Classifier(
            in_features=glimpses*image_features + question_features,
            hidden_features=1024,
            out_features=3000,
            dropout=0.5)

    def forward(self, img_features, question, q_len):
        question = self.text(question, q_len)
        img_features = self.attention(img_features, question)
        combined = torch.cat([img_features, question], dim=1)
        out = self.classifier(combined)
        return out  # returned output is not softmax-ed


class Attention(nn.Module):
    def __init__(self, image_features, question_features, mid_features, glimpses, dropout=0.0):
        super(Attention, self).__init__()

        self.image_conv = nn.Conv2d(
            image_features, mid_features, 1, bias=False)
        self.question_lin = nn.Linear(question_features, mid_features)
        self.attention_conv = nn.Conv2d(mid_features, glimpses, 1)
        self.drop = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, image, question):
        # image: batch x image_features x output_size x output_size
        # question: batch x question_features
        # batch x mid_features x output_size x output_size
        image_conv = self.image_conv(self.drop(image))
        question = self.question_lin(
            self.drop(question))  # batch x mid_features
        # batch x mid_features x output_size x output_size
        question = self.repeat_over_2d(question, image_conv)
        # batch x mid_features x output_size x output_size
        combined = image_conv + question
        combined = self.relu(combined)
        # batch x glimpses x output_size x output_size
        attention = self.attention_conv(self.drop(combined))

        n, c = image.size()[:2]
        s = image.size(2)*image.size(3)
        glimpses = attention.size(1)
        target_size = [n, glimpses, c, s]

        # batch x glimpses x output_size*output_size
        attention = attention.view(n, glimpses, -1)
        # batch*glimpses x output_size*output_size
        attention = attention.view(n * glimpses, -1)
        attention = F.softmax(attention, dim=1)

        # batch x image_features x output_size*output_size
        image = image.view(n, c, -1)
        image = image.view(n, 1, c, s).expand(*target_size)
        attention = attention.view(n, glimpses, 1, s).expand(*target_size)
        weighted = image * attention
        # batch x glimpses x image_features
        weighted_mean = weighted.sum(dim=3)
        out = weighted_mean.view(n, -1)  # batch x glimpses*image_features
        return out

    def repeat_over_2d(self, vector, target_vector):
        n, c = vector.size()
        spatial_size = target_vector.dim() - 2
        new_vector = vector.view(
            n, c, *([1] * spatial_size)).expand_as(target_vector)
        return new_vector


class Classifier(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, dropout=0.0):
        super(Classifier, self).__init__()
        self.drop1 = nn.Dropout(dropout)
        self.lin1 = nn.Linear(in_features, hidden_features)
        self.relu = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)
        self.lin2 = nn.Linear(hidden_features, out_features)

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
        self.embedding = nn.Embedding(
            embedding_tokens, embedding_features, padding_idx=0)
        self.drop = nn.Dropout(dropout)
        self.tanh = nn.Tanh()
        self.lstm = nn.LSTM(input_size=embedding_features,
                            hidden_size=lstm_features,
                            num_layers=1)
        self.features = lstm_features
        self._init(self.lstm, self.embedding)

    def _init(self, lstm, embedding):
        # lstm
        for w in lstm.weight_ih_l0.chunk(4, 0):
            init.xavier_uniform_(w)
        lstm.bias_ih_l0.data.zero_()

        for w in lstm.weight_hh_l0.chunk(4, 0):
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
