import json
import arghelper
import logging
import time
import torch
import numpy
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from utils import torch_helper, model_helper
from models import get_model
from data_helper import dthelper
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

logger = logging.getLogger(__name__)

START_TIMESTAMP = int(time.time())
print('---- TIMESTAMP: %s' % START_TIMESTAMP)
USE_CUDA = torch.cuda.is_available()
print('cuda is available? %s' % USE_CUDA)
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')


def infer(args):
    assert args.valid_file is not None
    valid_dict = dthelper.easy_get_matrix(args, args.valid_file, args.fields, memory_cache=True)
    valid_dataset = torch_helper.DictDataset(valid_dict, return_type='dict')
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    print('---- %s ---- dataset initialized' % time.asctime())

    # Init model and optimizer
    model = model_helper.load_model(args)

    torch.set_grad_enabled(False)
    model.eval()
    with torch.no_grad():
        preds = list()
        for dt in tqdm(valid_dataloader):
            dt = {k: v.to(DEVICE) for k, v in dt.items()}
            results = model(dt, is_train = False, is_valid = False)
            pred = results["pred"]
            preds.append(pred.detach().cpu().numpy())
        if args.output_path is not None:
            numpy.savetxt(args.output_path, preds, fmt='%s', delimiter='\t')

    print("done")  

if __name__ == "__main__":
    parser = arghelper.D2Parser()
    parser.add_argument("--fields", type=json, default=None)
    parser.add_argument("--valid_file", type=str, default=None)
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--eval_steps", type=int, default=500)
    
    parser.add_argument("--model_name", type=str, default = None)

    parser.add_argument("--output_path", type=str, default = None)

    args, _ = parser.parse_known_args()
    assert args.fields is not None
    assert args.load_model_path is not None

    if args.mode == "infer":
        infer(args)