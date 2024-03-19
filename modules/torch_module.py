import math
import torch
from torch import nn as nn
import torch.nn.functional as F


# USE_CUDA = torch.cuda.is_available()
USE_CUDA = False
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')


def reset_parameters(model, stdv):
    for m in model.modules():
        if isinstance(m, nn.Embedding):
            m.weight.data.uniform_(-stdv, stdv)
        elif isinstance(m, nn.Linear):
            m.weight.data.uniform_(-stdv, stdv)
        elif len([x.__class__.__name__ for x in m.children()]) > 0:
            for sub_m in m.children():
                reset_parameters(sub_m, stdv)


class L2Loss(nn.Module):
    def __init__(self):
        super(L2Loss, self).__init__()

    def forward(self, x):
        return torch.sum((x ** 2) / 2.0)


class Dice(nn.Module):
    def __init__(self, num_features, dim=3):
        super(Dice, self).__init__()
        assert dim == 2 or dim == 3
        # self.bn = nn.BatchNorm1d(num_features, eps=1e-9)
        self.sigmoid = nn.Sigmoid()
        self.dim = dim

        if self.dim == 3:
            self.alpha = nn.Parameter(torch.zeros((num_features, 1)))
        elif self.dim == 2:
            self.alpha = nn.Parameter(torch.zeros((num_features,)))

    def forward(self, x):
        if self.dim == 3:
            x = torch.transpose(x, 1, 2)
            # x_p = self.sigmoid(self.bn(x))
            x_p = self.sigmoid(x)
            out = self.alpha * (1 - x_p) * x + x_p * x
            out = torch.transpose(out, 1, 2)
        elif self.dim == 2:
            # x_p = self.sigmoid(self.bn(x))
            x_p = self.sigmoid(x)
            out = self.alpha * (1 - x_p) * x + x_p * x
        return out


class PositionalEncoder(nn.Module):
    def __init__(self, d_model, batch_first=True, dropout=0.1, max_len=1024):
        super(PositionalEncoder, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.batch_first = batch_first

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        if batch_first is True:
            pe = pe.unsqueeze(0)
        else:
            pe = pe.unsqueeze(1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        if self.batch_first is True:
            x = x + self.pe[:, :x.size(1)]
        else:
            x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class FullyConnectedLayer(nn.Module):
    def __init__(self, input_size, hidden_size, bias, batch_norm=False, dropout_rate=0.5, activation='relu', sigmoid=False, dice_dim=3):
        super(FullyConnectedLayer, self).__init__()
        assert len(hidden_size) >= 1 and len(bias) >= 1
        assert len(bias) == len(hidden_size)
        self.sigmoid = sigmoid

        layers = list()
        layers.append(nn.Linear(input_size, hidden_size[0], bias=bias[0]))

        for i, h in enumerate(hidden_size[:-1]):
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size[i]))

            if activation.lower() == 'relu':
                layers.append(nn.ReLU(inplace=True))
            elif activation.lower() == 'dice':
                assert dice_dim
                layers.append(Dice(hidden_size[i], dim=dice_dim))
            elif activation.lower() == 'prelu':
                layers.append(nn.PReLU())
            else:
                raise NotImplementedError

            layers.append(nn.Dropout(p=dropout_rate))
            layers.append(nn.Linear(hidden_size[i], hidden_size[i+1], bias=bias[i]))

        self.fc = nn.Sequential(*layers)
        if self.sigmoid:
            self.output_layer = nn.Sigmoid()

        # weight initialization xavier_normal (or glorot_normal in keras, tf)
        '''
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias.data)
        '''

    def forward(self, x):
        return self.output_layer(self.fc(x)) if self.sigmoid else self.fc(x)


class LocalActivationUnit(nn.Module):
    def __init__(self, hidden_size=[80, 40], bias=[True, True], embedding_dim=4, batch_norm=False):
        super(LocalActivationUnit, self).__init__()
        self.fc1 = FullyConnectedLayer(input_size=4*embedding_dim,
                                       hidden_size=hidden_size,
                                       bias=bias,
                                       batch_norm=batch_norm,
                                       activation='relu',
                                       dice_dim=3)

        self.fc2 = FullyConnectedLayer(input_size=hidden_size[-1],
                                       hidden_size=[1],
                                       bias=[True],
                                       batch_norm=batch_norm,
                                       activation='relu',
                                       dice_dim=3)

    def forward(self, query, user_behavior):
        # query ad            : size -> batch_size * 1 * embedding_size
        # user behavior       : size -> batch_size * time_seq_len * embedding_size

        user_behavior_len = user_behavior.size(-2)
        queries = torch.cat([query for _ in range(user_behavior_len)], dim=-2)

        attention_input = torch.cat([queries, user_behavior, queries-user_behavior, queries*user_behavior], dim=-1)
        attention_output = self.fc1(attention_input)
        attention_output = self.fc2(attention_output)

        return attention_output


class AttentionSequencePoolingLayer(nn.Module):
    def __init__(self, hidden_size=[64, 16], embedding_dim=4):
        super(AttentionSequencePoolingLayer, self).__init__()
        self.local_att = LocalActivationUnit(hidden_size=hidden_size, bias=[True, True], embedding_dim=embedding_dim, batch_norm=False)

    def forward(self, query_ad, user_behavior, mask=None):
        q_size = query_ad.size()
        u_size = user_behavior.size()
        if len(q_size) == 4:
            query_ad = query_ad.view(q_size[0]*q_size[1], q_size[2], q_size[3])
            user_behavior = user_behavior.view(u_size[0]*u_size[1], u_size[2], u_size[3])

        attention_score = self.local_att(query_ad, user_behavior)
        attention_score = torch.transpose(attention_score, 1, 2)  # B * 1 * T
        if mask is not None:
            attention_score = torch.mul(attention_score, mask.type(torch.FloatTensor))
        output = torch.matmul(attention_score, user_behavior)

        if len(q_size) == 4:
            output = output.view(q_size[0], q_size[1], output.size(-2), output.size(-1))

        return output


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()

    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0

        self.d_k = d_model // h
        self.h = h

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x)


class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = GELU()

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class TransformerBlock(nn.Module):
    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout):
        super().__init__()
        self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask=None):
        x = self.input_sublayer(x, lambda _x: self.attention.forward(
            _x, _x, _x, mask=mask
        ))
        x = self.output_sublayer(x, self.feed_forward)
        return self.dropout(x)
