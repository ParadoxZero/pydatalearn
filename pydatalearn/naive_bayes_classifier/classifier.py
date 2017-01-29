from pydatalearn.naive_bayes_classifier import summarizeByClass, predict


class NotLearnedError(Exception):
    pass


class Classifier:
    def __init__(self):
        self.summary = None
        self.__dataset = []

    def train(self, dataset):
        self.__dataset += dataset
        if len(self.__dataset) > 0:
            self.summary = summarizeByClass(self.__dataset)

    def predict(self, vector):
        if self.summary:
            return predict(self.summary, vector)
        raise NotLearnedError
