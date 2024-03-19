class BaseScoreBoarder(object):
    def __init__(self):
        self.labels = list()
        self.outputs = list()

    def init(self):
        return
    
    def clear(self):
        self.labels = list()
        self.outputs = list()

    def append(self, o, t):
        self.labels.append(t)
        self.outputs.append(o)

    def call_metric(self):
        raise NotImplementedError

    def log(self, prefix=None):
        return