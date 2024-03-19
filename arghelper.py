import os
import sys
import json
import torch
from data_helper import iohelper
from argparse import ArgumentParser


def boolean(x):
    if x is True:
        return True
    elif x is False:
        return False
    elif x.lower() == 'true':
        return True
    elif x.lower() == 'false':
        return False
    else:
        raise ValueError('Illegal boolean value')


def dictionary(x):
    if x is None or len(x) == 0:
        return dict()
    elif x.startswith('{') and x.endswith('}'):
        x = eval(x)
        assert isinstance(x, dict)
        return x
    else:
        d = dict()
        parts = x.split(',')
        for p in parts:
            k, v = p.split(':')
            k = k.strip('\'\"')
            v = eval(v)
            d[k] = v
        return d


def array(x):
    if x is None:
        return list()
    else:
        x = x.split(',')
        return x


def json_obj(x):
    if x is None:
        return json.loads('{}')
    elif x.startswith('file://'):
        path = x.replace('file://', '')
        with open(path, 'r') as f:
            return json.load(f)
    else:
        return json.loads(json.dumps(eval(x.strip('"').strip("'"))))


class BaseParser(ArgumentParser):
    def __init__(self, description='Common Base Parser'):
        super(BaseParser, self).__init__(description=description)

    def add_argument(self, *args, **kwargs):
        if kwargs.get('type', None) is bool:
            kwargs['type'] = boolean
        elif kwargs.get('type', None) is dict:
            kwargs['type'] = dictionary
            kwargs['default'] = "{}" if kwargs.get('default', None) is None else kwargs['default']
        elif kwargs.get('type', None) is list:
            kwargs['type'] = array
            kwargs['default'] = "[]" if kwargs.get('default', None) is None else kwargs['default']
        elif kwargs.get('type', None) is json:
            kwargs['type'] = json_obj
            kwargs['default'] = "{}" if kwargs.get('default', None) is None else kwargs['default']
        super(BaseParser, self).add_argument(*args, **kwargs)


class D2Parser(BaseParser):
    def __init__(self, description='General D2-Pai Arguments Parser'):
        super(D2Parser, self).__init__(description=description)

        # environment parameters
        self.add_argument("--mode", default='train', help="train, valid or predict, easy_train is available when distributed mode is off")
        self.add_argument("--environment", type=str, default='local', choices=['local', 'torch', 'tensorflow'],
                          help="run environment: local, torch or tensorflow")
        self.add_argument("--num_workers", type=int, default=1, help="num workers")
        self.add_argument("--worker_count", type=int, default=None, help="num workers")
        self.add_argument("--ps_hosts", type=str, default=None, help="num ps")
        self.add_argument("--worker_hosts", type=str, default=None, help="num workers")
        self.add_argument("--task_count", type=int, default=None, help="num workers")
        self.add_argument("--task_index", type=int, default=0, help="worker rank, need not to set this")
        self.add_argument("--device_id", type=int, default=None, help="gpu device id")
        self.add_argument("--job_name", type=str, default='worker', help="worker name, need not to set this")
        self.add_argument("--log_oss_path", type=str, help="log oss path", default=None)
        self.add_argument("--use_cache", type=bool, default='True', help="use data cache or not")
        self.add_argument("--label_key", type=str, default='label', help="label key")

        # model path parameters
        self.add_argument("--model_path", type=str, help="model state dict url", default=None)
        self.add_argument("--load_model_path", type=str, help="pretrained model state dict url", default=None)
        self.add_argument("--model_hdfs_path", type=str, help="model state dict hdfs url", default=None)
        self.add_argument("--checkpointDir", type=str, help="model state dict url", default=None)
        self.add_argument("--finetune_oss_path", type=str, help="fine-tuning model state dict url", default=None)
        self.add_argument("--config_oss_path", type=str, help="para config url, highest priority", default=None)

        # task type parameters
        self.add_argument("--task_type", type=str, help="class type, classification, regression or expectation")
        self.add_argument("--batch_size", type=int, default=None, help="batch size")
        self.add_argument("--score_type", type=str, default='loss', help="score type: f1, acc or loss")
        self.add_argument("--class_num", type=int, default=1, help="label class num")
        self.add_argument("--focus_class", type=int, default=None, help="focus on which output")
        self.add_argument("--id_key", type=str, default=None, help="id key")
        self.add_argument("--id_type", type=str, default='int', help="id type")
        self.add_argument("--learning_rate", type=float, default=1e-1, help="model lr")

        # input output parameters
        self.add_argument("--tables", type=str, help="train or predict tables")
        self.add_argument("--outputs", default=None, help="predict output tables")
        self.add_argument("--key_cols", type=str, default=None, help="inferred key columns")
        self.add_argument("--selected_cols", type=str, default='*', help="table selected cols")
        self.add_argument("--columns_type", type=str, default=None,
                          help="table selected cols type: id, image, text, float, entity, target or label")
        self.add_argument("--urlsafe", type=bool, default='False', help='image file type')
        self.add_argument("--start_cursor", type=int, default=0, help="start cursor")
        self.add_argument("--end_cursor", type=int, default=None, help="end cursor")
        self.add_argument("--total_epoch", type=int, default=1, help="training total epoch")
        self.add_argument("--part_size", type=int, default=10000, help="read table part size")
        self.add_argument("--allow_none", type=bool, default='False', help="allow None or not")
        self.add_argument("--file_delimiter", type=str, default='\t', help="file delimiter")
        self.add_argument("--pandas_delimiter", type=str, default=',', help="pandas data frame delimiter")
        self.add_argument("--hasher_type", type=str, default='tensorflow', choices=['tensorflow', 'sklearn', 'remainder'], help="hash mechanism, tf or sklearn")
        self.add_argument("--english_nlp", type=bool, default=False, help="chinese nlp or english nlp")
        self.add_argument("--hidden_dim", type=int, default=32, help="model hidden dim")
        self.add_argument("--emb_dim", type=int, default = 8, help="model embedding dim")
        self.add_argument("--optimizer", type=str, default='adam', help="optimizer")
        self.add_argument("--dropout", type=float, default=0.1, help="drop out rate")
        self.add_argument("--decay_rate", type=float, default=1e-5, help="model decay rate")

        self.add_argument("--scheduler_epoch", type=str, default=None, help="scheduler ecpoch")
        self.add_argument("--decay_epoch", type=int, default=None, help="decay epoch")

        # pandas paraeters
        self.add_argument("--header_num", type=int, default=0, help="pandas header num")


    def parse_known_args(self):
        args, unk = super(D2Parser, self).parse_known_args()

        if args.job_name != 'worker':
            args.task_count = len(args.worker_hosts.split(','))
            args.worker_count = len(args.worker_hosts.split(','))

        args.num_workers = args.task_count if args.task_count is not None else args.num_workers
        args.num_workers = args.worker_count if args.worker_count is not None else args.num_workers
        args.task_count = args.num_workers if args.task_count is None else args.task_count
        args.worker_count = args.num_workers if args.worker_count is None else args.worker_count
        assert args.num_workers == args.worker_count == args.task_count
        args.focus_class = args.class_num - 1 if args.focus_class is None else args.focus_class
        args.key_cols = args.selected_cols if args.key_cols is None else args.key_cols
        args.batch_size = 256 if args.batch_size is None else args.batch_size

        if args.job_name != 'worker' and 'train' not in args.mode:
            sys.exit()

        return args, unk

    def add_tf_flags(self, flags):
        args, unk = self.parse_known_args()
        for k, v in args.__dict__.items():
            if isinstance(v, bool):
                flags.DEFINE_boolean(k, v, '')
            elif isinstance(v, int):
                flags.DEFINE_integer(k, v, '')
            elif isinstance(v, float):
                flags.DEFINE_float(k, v, '')
            elif isinstance(v, str):
                flags.DEFINE_string(k, v, '')
            else:
                print('ignore converting args to flags - %s: %s' % (k, v))
