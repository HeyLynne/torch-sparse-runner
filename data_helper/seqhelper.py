import numpy
from . import hasher_helper


def filter_empty(seq, filter_zero=True):
    x = list()
    for k in seq:
        if k is not None and len(k) > 0 and (filter_zero is False or k != '0'):
            x.append(k)
    return x


def get_list_sequence(column):
    idx_matrix = list()
    time_matrix = list()
    for i in range(column.shape[0]):
        parts = column[i]
        parts = '' if parts is None else parts.replace(chr(29), ';')
        parts = parts.split(';')
        parts = [int(kv.split(':')[0]) for kv in parts]
        idx_matrix.append(parts)
    return idx_matrix, time_matrix


def get_str_list_sequence(column):
    idx_matrix = list()
    time_matrix = list()
    for i in range(column.shape[0]):
        parts = column[i]
        parts = '' if parts is None else parts.replace(chr(29), ';')
        parts = parts.split(';')
        parts = [kv.split(':')[0] for kv in parts]
        idx_matrix.append(parts)
    return idx_matrix, time_matrix


def get_kv_list_sequence(column):
    key_matrix = list()
    value_matrix = list()
    for i in range(column.shape[0]):
        parts = column[i]
        parts = '0:0' if parts is None else parts.replace(chr(29), ';')
        parts = parts.split(';')
        keys = [kv.split(':')[0] for kv in parts if len(kv) > 0]
        values = [int(kv.split(':')[1]) for kv in parts if len(kv) > 0]
        key_matrix.append(keys)
        value_matrix.append(values)
    return key_matrix, value_matrix


def get_list_multi_values(column, length):
    idx_matrix = list()
    for i in range(column.shape[0]):
        parts = column[i]
        parts = '' if parts is None else parts.replace(chr(29), ';')
        parts = parts.split(';')
        parts = [[int(kv.split(':')[j]) for j in range(length)] for kv in parts]
        idx_matrix.append(parts)
    return idx_matrix


def get_sequence(column, seq_length=32, padding_first=True, remain_last=True, sep=';'):
    idx_matrix = numpy.zeros([column.shape[0], seq_length], dtype=numpy.int64)
    time_matrix = numpy.zeros([column.shape[0], seq_length], dtype=numpy.int64)
    for i in range(column.shape[0]):
        parts = column[i]
        parts = '' if parts is None else parts.replace(chr(29), sep)
        parts = filter_empty(parts.split(sep))
        if len(parts) <= seq_length:
            parts = parts
        elif remain_last is True:
            parts = parts[-seq_length:]
        else:
            parts = parts[:seq_length]

        for j in range(len(parts)):
            kv = parts[j]
            if len(kv) > 0:
                kv = kv.split(':')
                k = int(kv[0])
                if padding_first is True:
                    p = seq_length-len(parts)+j
                    v = int(kv[1]) + j if len(kv) > 1 else j + 1
                    idx_matrix[i, p] = k
                    time_matrix[i, p] = v
                else:
                    v = int(kv[1]) + j if len(kv) > 1 else j + 1
                    idx_matrix[i, j] = k
                    time_matrix[i, j] = v

    return idx_matrix, time_matrix


def get_hasher_sequence(column, col_name, hasher_type='remainder', n_features=262144, seq_length=32, padding_first=True, sep=';'):
    if hasher_type == 'remainder':
        idx_list = numpy.zeros([column.shape[0] * seq_length], dtype=numpy.int64)
    else:
        idx_list = ['' for _ in range(column.shape[0] * seq_length)]
    time_list = [0 for _ in range(column.shape[0] * seq_length)]
    for i in range(column.shape[0]):
        parts = column[i]
        parts = '' if parts is None else parts.replace(chr(29), sep)
        parts = filter_empty(parts.split(sep))
        parts = parts if len(parts) <= seq_length else parts[-seq_length:]

        for j in range(len(parts)):
            kv = parts[j]
            if len(kv) > 0:
                kv = kv.split(':')
                if len(kv) == 1:
                    k = kv[0]
                    idx_list[i * seq_length + seq_length-len(parts)+j] = k
                    time_list[i * seq_length + seq_length-len(parts)+j] = 1
                else:
                    k = kv[0]
                    v = int(kv[1]) + j if len(kv) > 1 else j + 1
                    if padding_first is True:
                        idx_list[i * seq_length + seq_length-len(parts)+j] = k
                        time_list[i * seq_length + seq_length-len(parts)+j] = v
                    else:
                        idx_list[i * seq_length + j] = k
                        time_list[i * seq_length + j] = v

    idx_matrix, _ = hasher_helper.easy_get_hash_col(idx_list, hasher_type=hasher_type, col_name=col_name, n_features=n_features)
    idx_matrix = numpy.reshape(idx_matrix, [column.shape[0], seq_length])
    time_matrix = numpy.reshape(numpy.array(time_list, dtype=numpy.int64), [column.shape[0], seq_length])

    return idx_matrix, time_matrix
