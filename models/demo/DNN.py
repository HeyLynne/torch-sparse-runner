import torch
from ..BaseModel import BaseModel
from torch import nn as nn
import torch.nn.functional as F

class DNN(BaseModel):
    """
    self.linear_keys: sparse keys, size = embed_dim
    self.dense_keys: dense keys, size = dense_dim
    self.seq_keys: seq_keys
    """
    def __init__(self, args):
        super(DNN, self).__init__(args)
        self.label_key = args.label_key
        self.criterion = nn.BCELoss()
        # prepare model structure
        merge_dim = 0
        if len(self.linear_keys) > 0:
            self.linear_module = nn.Linear(self.embed_dim, args.hidden_dim)
            merge_dim += args.hidden_dim
        if len(self.dense_keys) > 0:
            self.dense_module = nn.Linear(self.dense_dim, args.hidden_dim)
            merge_dim += args.hidden_dim
        if len(self.seq_keys) > 0:
            self.seq_linear = nn.Linear(self.seq_dim, args.hidden_dim)
            self.seq_pe = PositionalEncoder(args.hidden_dim, batch_first=True, dropout=args.dropout)
            self.seq_module = torch.nn.TransformerEncoderLayer(
                args.hidden_dim, 4, args.hidden_dim, dropout=args.dropout, batch_first=True
            )
            merge_dim += args.hidden_dim
        
        self.batch_norm = nn.BatchNorm1d(merge_dim)
        self.merge_module = nn.Linear(merge_dim, args.hidden_dim)
        self.out_module = nn.Sequential(
            nn.Linear(args.hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def to_tensor(self, x_dict):
        keys = x_dict.keys()
        t_dict = dict()
        for k in keys:
            v = x_dict[k]
            if v.dtype == numpy.int64 or v.dtype == numpy.int32:
                t_dict[k] = torch.LongTensor(v).to(DEVICE)
            elif v.dtype == numpy.float32 or v.dtype == numpy.float64:
                t_dict[k] = torch.FloatTensor(v).to(DEVICE)
            else:
                raise TypeError("Illegal numpy dtype to dict tensor")
        return t_dict

    def forward(self, inputs, is_train = False, is_valid = False, score_boarder = None):
        e_list = list()
        d_list = list()
        for k in self.linear_keys:
            ek = self.embed_keys[k]
            x = inputs[k]
            e = F.dropout(self.embed_dict[ek](x), self.dropout, self.training)
            if len(e.shape) == 3 and e.shape[1] == 1:
                e = e.squeeze(1)
            elif len(e.shape) == 4:
                e = F.max_pool2d(e, (e.shape[-2], 1))
                e = e.squeeze(1).squeeze(1)
            else:
                e = torch.mean(e.squeeze(1), dim=1)
            e_list.append(e)
        for k in self.dense_keys:
            x = inputs[k]
            d_list.append(x)
        x_list = list()
        if len(e_list) > 0:
            e = torch.concat(e_list, dim=-1)
            x_list.append(self.linear_module(e))
        if len(d_list) > 0:
            d = torch.concat(d_list, dim=-1)
            x_list.append(self.dense_module(d))
        for k in self.seq_keys:
            x = inputs[k]
            x = x.transpose(1, 2)
            x = self.seq_pe(self.seq_linear(x))
            x = self.seq_module(x)
            x_list.append(x[:, -1])
        x = torch.concat(x_list, dim=-1)
        x = self.batch_norm(x)
        h = F.relu(self.merge_module(x))
        outs = self.out_module(h)

        if is_train:
            o = outs.squeeze(-1)
            label = inputs[self.label_key].to(torch.float32).squeeze(-1)
            loss = self.criterion(o, label)
            if score_boarder is not None:
                score_boarder.append(o.detach().cpu().numpy(), label.detach().cpu().numpy())
            results = {
                "loss": loss,
                "pred": o
            }
        elif is_valid:
            o = outs.squeeze(-1)
            label = inputs[self.label_key].squeeze(-1)
            if score_boarder is not None:
                score_boarder.append(o.detach().cpu().numpy(), label.detach().cpu().numpy())
            results = {
                "pred": o,
                "label": label
            }
        else:
            o = outs.squeeze(-1)
            results = {
                "pred": o
            }
        return results
