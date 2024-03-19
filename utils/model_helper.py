import torch
from models import get_model

def load_model(args):
    model = get_model(args.model_name)(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    if args.load_model_path is not None:
        print("Load model from %s" % args.load_model_path)
        model.load_state_dict(torch.load(args.load_model_path))
    return model
