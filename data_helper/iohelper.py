import sys
import numpy

python_version = (3 if sys.version.startswith('3') else 2)
folder_path = sys.path[0]
print(folder_path)
if python_version == 2:
    reload(sys)
    sys.setdefaultencoding("utf-8")


COUNT_CACHE = dict()

def easy_read_table(args_or_config, table_path, start=0, end=None, read_batch=1000, environment=None):
    if table_path.startswith('file://'):
        local_path = table_path.replace('file://', '')
        delimiter = args_or_config.file_delimiter if hasattr(args_or_config, 'file_delimiter') else ','
        # matrix = numpy.loadtxt(local_path, dtype='<U1000', delimiter=delimiter, encoding='utf-8')
        matrix = numpy.loadtxt(local_path, dtype=numpy.str, delimiter=delimiter, encoding='utf-8')
        matrix = matrix[start:end]
    elif table_path.startswith('pandas://'):
        import csv
        import pandas
        local_path = table_path.replace('pandas://', '')
        csv.field_size_limit(500 * 1024 * 1024)
        delimiter = args_or_config.file_delimiter if hasattr(args_or_config, 'file_delimiter') else ','
        header = args_or_config.header_num if hasattr(args_or_config, 'header_num') else None
        data_frame = pandas.read_csv(local_path, sep=delimiter, engine='python', header=header, dtype=str)
        matrix = data_frame.values
        matrix = matrix.astype(str)
        matrix = matrix[start:end]
    else:
        raise IOError('local only')
    return matrix


def easy_get_table_row_count(args_or_config, table_path):
    if COUNT_CACHE.get(table_path, None) is not None:
        return COUNT_CACHE.get(table_path)

    if table_path.startswith('file://') or table_path.startswith('pandas://'):
        local_path = table_path.replace('file://', '').replace('pandas://', '')
        with open(local_path, 'r') as f:
            lines = f.readlines()
            row_count = len(lines)
    else:
        raise IOError('local only')
    COUNT_CACHE[table_path] = row_count
    return row_count


def np2txt(values):
    if values.dtype == numpy.float32:
        values = numpy.round(values, 4)
    values = values.astype(numpy.str)
    if len(values.shape) == 1:
        texts = ','.join(values.tolist())
    elif len(values.shape) == 2:
        texts = '\n'.join([','.join(v.tolist()) for v in values])
    elif len(values.shape) == 3:
        texts = '\n'.join([';'.join([','.join(v.tolist()) for v in row]) for row in values])
    else:
        texts = '%s' % values
    return texts


def np2list(values):
    if values.dtype == numpy.float32:
        values = numpy.round(values, 4)
    if len(values.shape) == 1:
        outputs = values.tolist()
    elif len(values.shape) == 2:
        outputs = [v.tolist() for v in values]
    elif len(values.shape) == 3:
        outputs = [[v.tolist() for v in row] for row in values]
    else:
        raise ValueError("Illegal numpy matrix to list")
    return outputs
