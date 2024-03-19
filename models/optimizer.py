from torch import optim 

def init_optimizer(model, args):
    optimizer_type = args.optimizer
    learning_rate = args.learning_rate
    weight_decay = args.decay_rate
    if optimizer_type == "adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate,
                               weight_decay = weight_decay)
    elif optimizer_type == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate,
                             weight_decay = weight_decay)
    else:
        raise NotImplementedError

    return optimizer