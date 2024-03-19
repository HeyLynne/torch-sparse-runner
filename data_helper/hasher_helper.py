import numpy


hasher_dict = dict()


def easy_get_hash_col(col, hasher_type='tensorflow', col_name=None, n_features=1024):
    if hasher_type == 'sklearn':
        return get_hash_col(col, col_name, n_features)
    elif hasher_type == 'tensorflow':
        return get_tf_hash_col(col, col_name, n_features)
    elif hasher_type == 'remainder':
        return get_remainder_col(col, col_name, n_features)
    else:
        raise ValueError("Illegal hash mechanism")


def get_hash_col(col, col_name=None, n_features=1024):
    from sklearn.feature_extraction import FeatureHasher

    d_list = list()
    hasher = hasher_dict.setdefault(n_features, FeatureHasher(n_features=n_features, input_type='pair', dtype=numpy.int64, alternate_sign=True))
    for v in col:
        d_list.append([('%s_%s' % (col_name, v), 1)])
    indices = hasher.transform(d_list).indices
    indices = numpy.array(indices, dtype=numpy.int64)
    src_unique = numpy.unique(col).shape[0]
    tgt_unique = numpy.unique(indices).shape[0]
    collision = src_unique - tgt_unique

    if collision > 0:
        print('hasher feature col %s, %d hash feature num, collision num: %d (%d/%d)' % (col_name, n_features, collision, tgt_unique, src_unique))

    return indices, collision


def get_tf_hash_col(col, col_name=None, n_features=1024):
    import tensorflow
    if tensorflow.__version__.startswith('2'):
        import tensorflow.compat.v1 as tf
    else:
        import tensorflow as tf
    tf.reset_default_graph()
    graph = tf.Graph()
    col = ['%s_%s' % (col_name, v) for v in col]
    with tf.Session(graph=graph) as sess:
        hasher = tf.string_to_hash_bucket_fast(col, n_features, name=col_name)
        indices = sess.run(hasher)
    tf.reset_default_graph()
    src_unique = numpy.unique(col).shape[0]
    tgt_unique = numpy.unique(indices).shape[0]
    collision = src_unique - tgt_unique

    if collision > 0:
        print('hasher feature col %s, %d hash feature num, collision num: %d (%d/%d)' % (col_name, n_features, collision, tgt_unique, src_unique))

    return indices, collision


def get_remainder_col(col, col_name=None, n_features=1024):
    col[col == None] = 0
    col = col.astype(numpy.int64)
    col[col == 0] = 0
    indices = col % n_features
    indices = indices + 1

    indices = numpy.where(col > 0, indices, 0)

    src_unique = numpy.unique(col).shape[0]
    tgt_unique = numpy.unique(indices).shape[0]
    collision = src_unique - tgt_unique

    if collision > 0:
        print('hasher feature col %s, %d hash feature num, collision num: %d (%d/%d)' % (col_name, n_features, collision, tgt_unique, src_unique))

    return indices, collision


if __name__ == '__main__':
    get_remainder_col(numpy.array([
        [0, 1, 2],
        [1, 2, 3],
        [2, 3, 4],
        [3, 4, 0]
    ]))
