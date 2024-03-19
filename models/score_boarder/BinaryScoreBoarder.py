import numpy
import logging
from sklearn.metrics import confusion_matrix, recall_score, f1_score, accuracy_score
from .BaseScoreBoarder import BaseScoreBoarder


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

logger = logging.getLogger(__name__)

class BinaryScoreBoarder(BaseScoreBoarder):
    def __init__(self):
        self.labels = list()
        self.outputs = list()
    
    def calc_gini_coef(self, labels, probs):
        sort_order = []
        actual_cum = []
        for i in range(0, len(labels)):
            sort_order.append(i)
            actual_cum.append(0)
        sort_order = sorted(sort_order, key=lambda x: probs[x])
        inner_sum = 0.0
        actual_sum = 0.0
        actual_cum_sum = 0.0
        for i in range(0, len(labels)):
            current_label = labels[sort_order[i]]
            inner_sum += current_label
            actual_sum += current_label
            actual_cum[i] = inner_sum
            actual_cum_sum += actual_cum[i]
        gini_sum = actual_cum_sum / actual_sum - (len(labels) + 1) / 2.0
        gini_coef = gini_sum / len(labels)
        return gini_coef
    
    def cal_pre_and_recall(self):
        thresh = 0.1
        while thresh <= 0.7:
            preds = self.outputs >= thresh
            cm = confusion_matrix(self.labels, preds)
            acc = accuracy_score(self.labels, preds)
            rec, f1 = recall_score(self.labels, preds, average='binary'), f1_score(self.labels, preds, average='binary')
            print(f"--------------evaluate result thresh = {thresh}--------------")
            print("confusion matrix: ")
            print(cm)
            print(f"acc: {acc}\nrec: {rec}\nf1: {f1}")
            thresh += 0.05
        return
    
    def call_metric(self):
        self.labels = numpy.concatenate(self.labels)
        self.outputs = numpy.concatenate(self.outputs)
        self.cal_pre_and_recall()
        self.clear()

    def log(self, prefix=None):
        return