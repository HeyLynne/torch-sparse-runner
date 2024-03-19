import json
import arghelper
import logging
import time
import torch
import numpy
import os
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from utils import torch_helper, model_helper
from models import get_score_boarder
from models.optimizer import init_optimizer
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


def train(args):
    assert args.train_file is not None
    assert args.valid_file is not None
    train_dict = dthelper.easy_get_matrix(args, args.train_file, args.fields, memory_cache=True)
    train_dataset = torch_helper.DictDataset(train_dict, return_type='dict')
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)

    if args.do_eval:
        valid_dict = dthelper.easy_get_matrix(args, args.valid_file, args.fields, memory_cache=True)
        valid_dataset = torch_helper.DictDataset(valid_dict, return_type='dict')
        valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)
    
    if args.model_path is None:
        logger.warning("Model output path is null")
    
    if os.path.exists(args.model_path):
        logger.warning("Output path exists, check whether need to change a name of model")
    os.makedirs(args.model_path, exist_ok=True)

    print('---- %s ---- dataset initialized' % time.asctime())

    # Init model and optimizer
    model = model_helper.load_model(args)

    optimizer = init_optimizer(model, args)

    if args.scheduler_epoch is not None:
        scheduler_step = [int(step) for step in args.scheduler_epoch.split(',')]
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=scheduler_step, gamma=args.decay_rate)
    elif args.decay_epoch is not None:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.decay_epoch, gamma=args.decay_rate)
    else:
        lr_scheduler = None

    # train model
    train_boarder = None
    valid_boarder = None
    if args.init_score_boarder:
        print(args.boarder_name)
        train_boarder = get_score_boarder(args.boarder_name)()
        train_boarder.init()
        if args.do_eval:
            valid_boarder = get_score_boarder(args.boarder_name)()
            valid_boarder.init()
    global_step = 0
    for e in range(args.total_epoch):
        print('---- %s ---- training at epoch %d' % (time.asctime(), e))
        torch.set_grad_enabled(True)
        model.train()
        total_loss = 0

        step = 0
        for dt in tqdm(train_dataloader):
            dt = {k: v.to(DEVICE) for k, v in dt.items()}
            optimizer.zero_grad()
            results = model(dt, is_train = True, score_boarder = train_boarder)
            loss, pred = results["loss"], results["pred"]
            if args.do_summary:
                writer.add_scalar(f'train/loss', loss, global_step=global_step)
            total_loss += loss
            loss.backward()
            optimizer.step()
            step += 1
            global_step += 1
        print(f"epoch {e}, average loss: {total_loss} / {step}")
        if args.do_eval:
            torch.set_grad_enabled(False)
            model.eval()
            with torch.no_grad():
                labels = list()
                preds = list()
                for dt in tqdm(valid_dataloader):
                    dt = {k: v.to(DEVICE) for k, v in dt.items()}
                    results = model(dt, is_train = False, is_valid = True, score_boarder = valid_boarder)
                    pred, label = results["pred"], results["label"]
                    preds.append(pred.detach().cpu().numpy())
                    labels.append(label.detach().cpu().numpy())
                if args.output_path is not None:
                    pred_results = numpy.expand_dims(numpy.concatenate(preds), axis=1).astype(numpy.str)
                    label_results = numpy.expand_dims(numpy.concatenate(labels), axis=1).astype(numpy.str)
                    mat = numpy.concatenate([pred_results, label_results], axis=1)
                    numpy.savetxt(args.output_path, mat, fmt='%s', delimiter='\t')
        
        if train_boarder is not None:
            print("----train---")
            train_boarder.call_metric()
            train_boarder.log()
            train_boarder.clear()
        if valid_boarder is not None:
            print("----valid---")
            valid_boarder.call_metric()
            valid_boarder.log()
            valid_boarder.clear()
        
        if lr_scheduler is not None:
            lr_scheduler.step()
        if args.model_path is not None:
            file_path = os.path.join(args.model_path, "%s.%d" % (args.model_name, e))
            torch.save(model.state_dict(), file_path)

    print("done")  

if __name__ == "__main__":
    parser = arghelper.D2Parser()
    parser.add_argument("--fields", type=json, default=None)
    parser.add_argument("--train_file", type=str, default=None)
    parser.add_argument("--valid_file", type=str, default=None)

    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--model_name", type=str, default = None)
    parser.add_argument("--boarder_name", type=str, default = None)

    parser.add_argument("--do_eval", type=bool, default = False)
    parser.add_argument("--init_score_boarder", type=bool, default = False)
    parser.add_argument("--output_path", type=str, default = None)

    parser.add_argument("--do_summary", type=bool, default = False)
    parser.add_argument("--summary_path", type=str, default = None)
    

    args, _ = parser.parse_known_args()
    assert args.fields is not None

    if args.init_score_boarder:
        assert args.boarder_name is not None
        logger.info("score boarder name: %s" % (args.boarder_name))
    if args.do_eval:
        assert args.valid_file is not None

    if args.do_summary:
        assert args.summary_path is not None
        writer = SummaryWriter(args.summary_path)

    if args.mode == "train":
        train(args)