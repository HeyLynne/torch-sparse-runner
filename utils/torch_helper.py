import numpy
import scipy
import torch
from torch.utils.data import Dataset


class KVDataset(Dataset):
    def __init__(self, keys, dtypes, values, return_type='tuple'):
        super(KVDataset, self).__init__()

        assert isinstance(keys, tuple) or isinstance(keys, list)

        dtypes = [dtypes[k] for k in keys] if isinstance(dtypes, dict) else dtypes
        values = [values[k] for k in keys] if isinstance(values, dict) else values

        assert isinstance(dtypes, tuple) or isinstance(dtypes, list)
        assert isinstance(values, tuple) or isinstance(values, list)
        assert return_type in ('tuple', 'dict')

        self.keys = list()
        self.d_list = list()
        self.dtypes = list()
        self.return_type = 0 if return_type == 'tuple' else 1

        for i in range(len(values)):
            if values[i] is not None and len(values[i].shape) > 0:
                if isinstance(values[i], scipy.sparse.spmatrix):
                    self.keys.append(keys[i])
                    '''
                    # todo, >= 1.4 version torch is needed
                    if dtypes[i] == torch.int64:
                        self.d_list.append(
                            torch.sparse.LongTensor(
                                torch.LongTensor(values[i].nonzero()),
                                torch.LongTensor(values[i].data),
                                torch.Size(values[i].shape),
                            )
                        )
                    else:
                        self.d_list.append(
                            torch.sparse.FloatTensor(
                                torch.LongTensor(values[i].nonzero()),
                                torch.FloatTensor(values[i].data),
                                torch.Size(values[i].shape),
                            )
                        )
                    '''
                    self.d_list.append(values[i])
                    self.dtypes.append(dtypes[i])
                elif isinstance(values[i], numpy.ndarray):
                    self.keys.append(keys[i])
                    self.d_list.append(torch.tensor(values[i], dtype=dtypes[i]))
                    self.dtypes.append(dtypes[i])

    def set_return(self, return_type):
        assert return_type in ('tuple', 'dict')
        self.return_type = 0 if return_type == 'tuple' else 1

    def to_dict(self, dt):
        assert len(self.keys) == len(dt)
        d_dict = dict()
        for i, k in enumerate(self.keys):
            d_dict[k] = dt[i]
        return d_dict

    def to_tuple(self, dt):
        d_tuple = [None for _ in self.keys]
        for i, k in enumerate(self.keys):
            d_tuple[i] = dt[k]
        return tuple(d_tuple)

    def __getitem__(self, item):
        if self.return_type == 0:
            d_tuple = list()
            for i in range(len(self.keys)):
                if isinstance(self.d_list[i], scipy.sparse.spmatrix):
                    d_tuple.append(torch.tensor(self.d_list[i][item].dense(), dtype=self.dtypes[i]))
                else:
                    d_tuple.append(self.d_list[i][item])
            d_tuple = tuple(d_tuple)
            return d_tuple
        else:
            d_dict = dict()
            for i in range(len(self.keys)):
                if isinstance(self.d_list[i], scipy.sparse.spmatrix):
                    d_dict[self.keys[i]] = torch.tensor(self.d_list[i][item].toarray().squeeze(0), dtype=self.dtypes[i])
                else:
                    d_dict[self.keys[i]] = self.d_list[i][item]
            return d_dict

    def __len__(self):
        return self.d_list[0].size(0)


class DictDataset(KVDataset):
    def __init__(self, x_dict, return_type='tuple'):
        keys = sorted([k for k, _ in x_dict.items()])
        dtypes = list()
        values = list()
        for k in keys:
            v = x_dict[k]
            values.append(v)
            if v.dtype == numpy.int64 or v.dtype == numpy.int32:
                dtypes.append(torch.int64)
            elif v.dtype == numpy.float32 or v.dtype == numpy.float64:
                dtypes.append(torch.float32)
            else:
                raise TypeError("Illegal numpy dtype to dict dataset")
        super(DictDataset, self).__init__(keys, dtypes, values, return_type)
