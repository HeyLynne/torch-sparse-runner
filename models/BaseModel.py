import math
import logging
from torch import nn as nn

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

logger = logging.getLogger(__name__)

class BaseModel(nn.Module):
    def __log_infos(self):
        dense_keys = ",".join(self.dense_keys)
        sparse_keys = ",".join(self.linear_keys)
        seq_keys = ",".join(self.seq_keys)
        logger.info("dense keys: %s" % dense_keys)
        logger.info("sparse_keys: %s" % sparse_keys)
        logger.info("seq_keys: %s" % seq_keys)
        logger.info("Sparse dim: %d, Dense dim: %d, Seqence dim: %d"% (self.embed_dim, self.dense_dim, self.seq_dim))

    def __init__(self, args):
        super().__init__()
        self.hidden_size = args.hidden_dim
        self.dropout = args.dropout
        self.emb_dim = args.emb_dim
        stdv = 1.0 / math.sqrt(self.hidden_size)

        self.dense_keys = list()
        self.linear_keys = list()
        self.seq_keys = list()
        self.embed_keys = dict()

        self.seq_dim = 0
        self.dense_dim = 0
        self.embed_dim = 0
        self.embed_dict = dict()

        for field in args.fields:
            gk = field.get('group_key', field['key'])
            embed_key = field.get('embed_key', gk)
            dtype = field.get('type', None)
            if dtype in ('sparse', 'sparse_seq') or field.get('float2entity', False) is True:
                self.embed_keys[gk] = embed_key
                if gk not in self.linear_keys:
                    self.linear_keys.append(gk)
                self.embed_dim += self.emb_dim
                if embed_key not in self.embed_dict:
                    embedding = nn.Embedding(field['i_dim'], self.emb_dim)
                    embedding.weight.data.uniform_(-stdv, stdv)
                    self.embed_dict[embed_key] = embedding
            elif dtype == 'dense':
                if gk not in self.dense_keys:
                    self.dense_keys.append(gk)
                self.dense_dim += field.get('length', 1)
        self.__log_infos()