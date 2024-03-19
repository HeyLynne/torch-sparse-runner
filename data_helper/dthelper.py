import math
import time
import numpy
import scipy
import logger
import random
from . import iohelper
from . import seqhelper
from . import hasher_helper
from scipy.sparse import lil_matrix
import warnings
import importlib
#from matplotlib import pyplot as plt
warnings.filterwarnings("ignore")


MEMORY_CACHE = dict()
COL_TYPE_SET = set(['id', 'image', 'text', 'dense', 'sparse', 'kv_dense', 'kv_sparse', 'sparse_seq', 'target', 'label', 'pass'])

def float2entity(col):
    col = col.astype(numpy.float32)
    col[col == None] = 0
    col = numpy.where(col <= 0, 0, col)
    col = numpy.log10(1 + col)
    col = col * 128
    result = numpy.floor(col)
    result = numpy.where(result >= 1024, 1023, result)
    result = numpy.expand_dims(result, axis=1).astype(numpy.int64)
    return result


def get_multi_sparse_features(column, separator):
    column = column.astype(numpy.str)
    column = numpy.char.split(column, sep=separator)
    column = numpy.array(column.tolist(), dtype=numpy.int64)
    return column

def call_method_of_trans(trans, x):
    module_name, func_name = trans.split(".")
    try:
        module = importlib.import_module(f"funcs.{module_name}")
        func = getattr(module, func_name)
        return func(x)
    except ImportError:
        import traceback
        traceback.print_exc()
        print(f"Module '{module_name}' not found.")
        return numpy.zeros_like(x)
    except AttributeError:
        print(f"Function '{func_name}' not found in module '{module_name}'.")
        return numpy.zeros_like(x)

def get_float_features(column, is_array=True, i_dim=None, separator=None, allow_none=True):
    if is_array:
        column = numpy.char.strip(column, '"\'')
        legal = numpy.logical_and(column != None, column != '')
        if i_dim is not None and separator is not None:
            float_features = numpy.zeros([column.shape[0], i_dim], dtype=numpy.float32)
            not_none = column[legal]
            if not_none.shape[0] > 0:
                not_none = not_none.astype(numpy.str)
                not_none = numpy.char.split(not_none, sep=separator)
                not_none = numpy.array(not_none.tolist(), dtype=numpy.float32)
                float_features[legal] = not_none
        else:
            not_none = column[legal]
            if separator is None:
                if ';' in not_none[0]:
                    separator = ';'
                elif ',' in not_none[0]:
                    separator = ','
                elif ' ' in not_none[0]:
                    separator = ' '
                else:
                    raise ValueError("Not separator in array features")
            not_none = not_none.astype(numpy.str)
            not_none = numpy.char.split(not_none, sep=separator)

            not_none = numpy.array(not_none.tolist(), dtype=numpy.float32)
            # not_none = numpy.where(numpy.isnan(not_none), numpy.zeros_like(not_none), not_none)

            float_features = numpy.zeros([column.shape[0], not_none.shape[1]], dtype=numpy.float32)
            float_features[legal] = not_none
    else:
        if allow_none is True:
            column = numpy.char.strip(column, '"\'')
            column[column == None] = 0
            column[column == ''] = 0
        column = column.astype(numpy.float32)
        float_features = numpy.expand_dims(column, axis=1)
    return float_features


def get_kv_dense(column, num):
    def handle(parts):
        parts = '' if parts is None else parts.replace(chr(29), ';')
        parts = seqhelper.filter_empty(parts.split(';'))

        arr = numpy.zeros([num], dtype=numpy.float32)
        for j in range(len(parts)):
            kv = parts[j]
            if len(kv) > 0:
                kv = kv.split(':')
                k = int(kv[0])
                v = float(kv[1])
                arr[k] = v
        return arr

    matrix = numpy.vstack(map(handle, column))
    return matrix


def filter_matrix(mat, col_idx, filter_cond):
    logger.log('row num before filtering: %d' % mat.shape[0])
    col = get_float_features(mat[:, col_idx], is_array=False, allow_none=True).flatten()
    if filter_cond.startswith('>='):
        mat = mat[col >= float(filter_cond.replace('>=', ''))]
    elif filter_cond.startswith('<='):
        mat = mat[col <= float(filter_cond.replace('<=', ''))]
    elif filter_cond.startswith('>'):
        mat = mat[col > float(filter_cond.replace('>', ''))]
    elif filter_cond.startswith('<'):
        mat = mat[col < float(filter_cond.replace('<', ''))]
    elif filter_cond.startswith('='):
        mat = mat[col == float(filter_cond.replace('=', ''))]
    else:
        raise ValueError('Illegal filter condition')
    logger.log('row num after filtering: %d' % mat.shape[0])
    return mat

def trans_number(x, trans=None):
    if trans is None:
        return x
    elif trans == 'ln':
        return numpy.log(1+numpy.where(x < 0, numpy.zeros_like(x), x))
    elif trans == 'log2':
        return numpy.log2(1+numpy.where(x < 0, numpy.zeros_like(x), x))
    elif trans == 'log10':
        return numpy.log10(1+numpy.where(x < 0, numpy.zeros_like(x), x))
    elif trans.startswith('div'):
        foot = float(trans.replace('div', ''))
        return x / foot
    elif trans.startswith('clip'):
        parts = [float(v) for v in trans.split(',')[1:]]
        return numpy.clip(x, parts[0], parts[-1])
    elif ',' in trans:
        parts = [float(v) for v in trans.split(',')]
        trans_values = numpy.zeros_like(x)
        for i in range(len(parts)):
            if i < len(parts) - 1:
                trans_values = numpy.where(
                    numpy.logical_and(x >= parts[i], x < parts[i+1]), i, trans_values
                )
            else:
                trans_values = numpy.where(x >= parts[i], i, trans_values)
        return trans_values
    else:
        return call_method_of_trans(trans, x)

def get_kv_sparse(column, num, trans=None):
    def handle(parts):
        parts = '' if parts is None else parts.replace(chr(29), ';')
        parts = seqhelper.filter_empty(parts.split(';'))

        row = list()
        col = list()
        values = list()
        for j in range(len(parts)):
            kv = parts[j]
            if len(kv) > 0:
                kv = kv.split(':')
                k = int(kv[0])
                v = float(kv[1])
                row.append(0)
                col.append(k)
                values.append(v)
        coo = scipy.sparse.coo_matrix((values, (row, col)), shape=(1, num))
        return coo

    column = numpy.char.strip(column, '"\'')
    matrix = scipy.sparse.vstack(map(handle, column))
    print(matrix.toarray())

    matrix = matrix.tocsr()
    if trans is not None:
        matrix = scipy.sparse.csr_matrix((trans_number(matrix.data, trans), matrix.indices, matrix.indptr), shape=matrix.shape)
    return matrix


def get_field_attr(fields, pkey, skey):
    for field in fields:
        if field['key'] == pkey or field.get('group_key', None) == pkey:
            return field.get(skey, None)
    return None


def get_field_matrix(args_or_config, table_path, fields, start=0, end=None, environment='local'):
    keys = [d['key'] for d in fields]
    selected_cols = ','.join(keys)
    print('selected cols: %s' % selected_cols)
    matrix = iohelper.easy_read_table(args_or_config, table_path, start=start, end=end, environment=environment)
    return matrix


def parse_field_matrix(matrix, fields, log=False):
    """
    :param matrix:
    [
    0, '1,2,3,4,5', 'bytes'
    ]
    :param fields:
    :param log:
    :return:
    """
    assert matrix.shape[1] == len(fields)

    x_dict = {field.get('group_key', field['key']): list() for field in fields if field['type'] != 'pass'}

    for j, field in enumerate(fields):
        filter_cond = field.get('filter', None)
        if filter_cond is not None:
            for cond in filter_cond.split(','):
                matrix = filter_matrix(matrix, j, cond)

    for j, field in enumerate(fields):
        col = matrix[:, j]
        # key name
        k = field['key']
        # type name
        t = field['type']
        gk = field.get('group_key', k)
        offset = field.get('offset', 0)

        assert t in COL_TYPE_SET
        print(f"processing key {k}")
        if t == 'id':
            x = col.astype(numpy.int64)
            x = numpy.expand_dims(x, axis=-1)
            x_dict[gk].append(x + offset)
        elif t == 'dense':
            length = field.get('length', 0)
            if length > 0:
                sep = field.get('sep', None)
                i_dim = field.get('length', None)
                rng = field.get('range', None)
                x = get_float_features(col, is_array=True, i_dim=i_dim, allow_none=True, separator=sep)
                assert x.shape[1] == length
                trans = field.get('trans', None)
                x = trans_number(x, trans=trans)
                if rng is not None:
                    rng = [int(v) for v in rng.split(',')]
                    x[:, rng[0]:rng[1]] = 0
                x_dict[gk].append(x + offset)
            else:
                x = get_float_features(col, is_array=False, allow_none=True)
                x = x.astype(numpy.float32)
                trans = field.get('trans', None)
                x = trans_number(x, trans)
                if field.get('float2entity', False) is True:
                    x_dict[gk].append(float2entity(col) + offset)
                else:
                    x_dict[gk].append(x + offset)
        elif t == 'sparse':
            hash_type = field.get('hash_type', None)
            if hash_type is None:
                col[col == None] = 0
                x = col.astype(numpy.int64)
            elif hash_type == 'string':
                x = numpy.expand_dims(col.astype(numpy.str), axis=-1)
            else:
                n_features = field['i_dim']
                x, collision = hasher_helper.easy_get_hash_col(col, hash_type, n_features=n_features, col_name=field.get('embed_key', gk))
                x = numpy.expand_dims(x, axis=1)
            x = numpy.expand_dims(x, axis=-1)
            if hash_type != 'string':
                x_dict[gk].append(x + offset)
            else:
                x_dict[gk].append(x)
        elif t == 'text':
            logger.log('ignored, text not support yet')
            '''
            seq_length = field.get('seq_length', 64)
            x = text_helper.get_seq_matrix(col, crop_head=True, crop_tail=False, args=config, seq_length=seq_length)
            x = numpy.expand_dims(x, axis=1)
            x_dict[gk].append(x + offset)
            '''
        elif t == 'sparse_seq':
            hash_type = field.get('hash_type', None)
            padding_first = field.get('padding_first', False)
            remain_last = field.get('remain_last', True)
            seq_length = field.get('seq_length', 32)
            if hash_type is None or hash_type == 'string':
                x, mask = seqhelper.get_sequence(col, seq_length=seq_length, padding_first=padding_first, remain_last=remain_last)
                x = numpy.expand_dims(x.astype(numpy.int64), axis=1)
                mask = numpy.expand_dims(mask, axis=1)
                x_dict[gk].append(x + offset)
            elif hash_type == 'list':
                x, _ = seqhelper.get_list_sequence(col)
                x_dict[gk].append(x)
            elif hash_type == 'str_list':
                x, _ = seqhelper.get_str_list_sequence(col)
                x_dict[gk].append(x)
            elif hash_type == 'kv_list':
                x, v = seqhelper.get_kv_list_sequence(col)
                x_dict[gk].append((x, v))
            else:
                n_features = field['i_dim']
                sep = field.get('sep', ';')
                x, mask = seqhelper.get_hasher_sequence(
                    col, col_name=k, hasher_type=hash_type, seq_length=seq_length, n_features=n_features,
                    padding_first=padding_first, sep=sep
                )
                x = numpy.expand_dims(x, axis=1)
                mask = numpy.expand_dims(mask, axis=1)
                x_dict[gk].append(x + offset)
        elif t == 'label':
            trans = field.get('trans', None)
            x = col
            if trans is not None:
                x = x.astype(numpy.float32)
                x = trans_number(x, trans)
            x = x.astype(numpy.int64)
            x = numpy.expand_dims(x, axis=-1)
            x_dict[gk].append(x + offset)
        elif t == 'target':
            trans = field.get('trans', None)
            x = get_float_features(col, is_array=False, allow_none=True).flatten()
            if trans is not None:
                x = x.astype(numpy.float32)
                x = trans_number(x, trans)
            x = x.astype(numpy.float32)
            x = numpy.expand_dims(x, axis=-1)
            x_dict[gk].append(x + offset)
        elif t == 'pass':
            pass
        else:
            raise ValueError("Column Type [ %s ] Not Allowed" % t)
        if log is True:
            logger.log('get col %s, type %s' % (k, t))
            logger.log('ori data case -- %s' % ('%s' % col[0]).replace('\n', '\t').replace('  ', ' ')[:128])
            logger.log('use data case -- %s' % ('%s' % x[0]).replace('\n', '\t').replace('  ', ' ')[:128])
    for k, v in x_dict.items():
        if isinstance(v[0], scipy.sparse.spmatrix):
            x_dict[k] = scipy.sparse.hstack(v).tocsr()
        elif isinstance(v[0], list):
            assert len(v) == 1
            x_dict[k] = v[0]
        else:
            x_dict[k] = numpy.concatenate(v, axis=1)
    return x_dict


def easy_get_matrix(args_or_config, table_path, fields, start=0, end=None, environment='local', use_cache=True, memory_cache=False, parse_when_read=False, log=False):
    if use_cache is False:
        matrix = get_field_matrix(args_or_config, table_path, fields, start=start, end=end, environment=environment)
        x_dict = parse_field_matrix(matrix, fields, log=log)
        return x_dict
    else:
        cache_path = '%s_%s' % (table_path.replace('cache://', '').replace('/', '_'), '%s_%s.npz' % (start, end))
        x_dict = MEMORY_CACHE[cache_path] if cache_path in MEMORY_CACHE else None
        if x_dict is None:
            matrix = get_field_matrix(args_or_config, table_path, fields, start=start, end=end, environment=environment)
            print(f"----load data finish, matrix shape: {matrix.shape}, fields length: {len(fields)}")
            # if matrix.shape[1] > len(fields):
            #     matrix = matrix[:, :len(fields)]
            print('-- %s -- read %s done' % (time.asctime(), table_path))

            x_dict = parse_field_matrix(matrix, fields, log=log)
            print('-- %s -- parse %s done' % (time.asctime(), table_path))
            if memory_cache is True:
                MEMORY_CACHE[cache_path] = x_dict
            return x_dict
        else:
            print('-- %s -- load %s [%s, %s] from cache' % (time.asctime(), table_path, start, end))
            return x_dict


def get_group_shape(input_dict, group_fields=None):
    import tensorflow as tf
    group_keys = sorted([k for k, _ in group_fields.items()]) if group_fields is not None else sorted([k for k, _ in input_dict.items()])
    group_types = dict()
    group_shapes = dict()
    hash_keys = set([k for k, f in group_fields if f.get('hash_type', None) == 'string']) if group_fields is not None else set()
    if group_fields is not None:
        field_shapes = {k: [int(x) for x in group_fields[k]['shape'].split(',')] for k in group_keys}
    else:
        field_shapes = {k: v.shape for k, v in input_dict.items()}
    for k in group_keys:
        v = input_dict[k]
        if k in hash_keys:
            group_types[k] = tf.string
        elif v.dtype == numpy.ones([1], dtype=numpy.int64).astype(numpy.str).dtype:
            group_types[k] = tf.string
        elif v.dtype == numpy.int32 or v.dtype == numpy.int64:
            group_types[k] = tf.int64
        elif v.dtype == numpy.float64 or v.dtype == numpy.float32 or v.dtype == numpy.float16:
            group_types[k] = tf.float32
        else:
            group_types[k] = tf.string
            # raise TypeError("Illegal numpy dtype to tf dtype")
        group_shapes[k] = tf.TensorShape(field_shapes[k][1:])
    return group_keys, group_types, group_shapes


def easy_np2tf_dataset(args_or_config, table_path, fields, group_fields, cursor_list, environment, label_key='label'):
    import tensorflow as tf

    input_dict = easy_get_matrix(args_or_config, table_path, fields, start=cursor_list[0][0], end=cursor_list[0][1], log=True, environment=environment)
    group_keys, group_types, group_shapes = get_group_shape(input_dict, group_fields)

    def np_generator():
        for start, end in cursor_list:
            input_dict = easy_get_matrix(args_or_config, table_path, fields, start=start, end=end, log=True, environment=environment)
            data_len = input_dict[group_keys[0]].shape[0]

            for i in range(data_len):
                f = dict()
                for k in group_keys:
                    v = input_dict[k][i]
                    if isinstance(v, numpy.ndarray):
                        v = v.reshape(group_shapes[k])
                        f[k] = v
                    elif isinstance(v, scipy.sparse.spmatrix):
                        v = numpy.squeeze(numpy.asarray(v.todense()), axis=0)
                        v = v.reshape(group_shapes[k])
                        f[k] = v
                    else:
                        raise TypeError("Illegal dtype to tf dtype")
                l = f[label_key]
                del f[label_key]
                yield (f, l)

    def copy_exclude(d, e_k):
        o = dict()
        for k, v in d.items():
            if k != e_k:
                o[k] = v
        return o

    dataset = tf.data.Dataset.from_generator(
        np_generator,
        (copy_exclude(group_types, label_key), group_types[label_key]),
        (copy_exclude(group_shapes, label_key), group_shapes[label_key])
    )
    return dataset


def easy_np_dict_dataset(input_dict, label_key='label'):
    import tensorflow as tf

    group_keys, group_types, group_shapes = get_group_shape(input_dict)

    def np_generator():
        data_len = input_dict[group_keys[0]].shape[0]
        indices = [i for i in range(data_len)]
        for i in indices:
            f = dict()
            for k in group_keys:
                v = input_dict[k][i]
                if isinstance(input_dict[k], numpy.ndarray):
                    v = v.reshape(group_shapes[k])
                    f[k] = v
                elif isinstance(v, scipy.sparse.spmatrix):
                    v = numpy.squeeze(numpy.asarray(v.todense()), axis=0)
                    v = v.reshape(group_shapes[k])
                    f[k] = v
                else:
                    raise TypeError("Illegal dtype to tf type")
            if label_key is not None:
                l = f[label_key]
                del f[label_key]
                yield (f, l)
            else:
                yield (f, numpy.array([0], dtype=numpy.int64))

    def copy_exclude(d, e_k):
        o = dict()
        for k, v in d.items():
            if k != e_k:
                o[k] = v
        return o

    if label_key is not None:
        dataset = tf.data.Dataset.from_generator(
            np_generator,
            (copy_exclude(group_types, label_key), group_types[label_key]),
            (copy_exclude(group_shapes, label_key), group_shapes[label_key])
        )
    else:
        dataset = tf.data.Dataset.from_generator(
            np_generator,
            (group_types, tf.int64),
            (group_shapes, tf.TensorShape([1]))
        )
    return dataset


class BatchGenerator(object):
    def __init__(self, size, batch, indices=None, max_batch_num=None, repeat=True):
        self.indices = [i for i in range(size)] if indices is None else indices
        self.shuffle()
        self.size = size
        self.repeat = repeat
        assert size == len(self.indices)
        self.batch = batch
        self.batch_num = len(self.indices) // batch
        if max_batch_num is not None:
            self.batch_num = min(self.batch_num, max_batch_num)
        self.batch_cursor = 0

    def shuffle(self):
        random.shuffle(self.indices)

    def get(self):
        if self.batch_cursor >= self.batch_num:
            if self.repeat is True:
                self.shuffle()
                self.batch_cursor = 0
            else:
                return None
        b = self.indices[self.batch_cursor*self.batch:self.batch_cursor*self.batch+self.batch]
        self.batch_cursor += 1
        return b

