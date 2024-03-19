from .score_boarder.BinaryScoreBoarder import BinaryScoreBoarder

from .demo.DNN import DNN

model_list = {
    "DNN": DNN
}

score_boarder_list = {
    "BinaryScoreBoarder": BinaryScoreBoarder
}

def get_model(model_name):
    if model_name in model_list.keys():
        model = model_list[model_name]
        return model
    else:
        raise NotImplementedError

def get_score_boarder(score_boarder_name):
    if score_boarder_name in score_boarder_list.keys():
        return score_boarder_list[score_boarder_name]
    else:
        print(f"Error : {score_boarder_name} not exist")
        return None