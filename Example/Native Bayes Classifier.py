"""
Dataset description:
https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.names
"""
from pydatalearn import naive_bayes_classifier as nb
from pydatalearn import splitDataSet,loadCSV
from pydatalearn.naive_bayes_classifier.classifier import Classifier


def main():
    print("Loading pima-indians-diabetes.data.csv...")
    data_set = (loadCSV("DataSet/pima-indians-diabetes.data.csv"))
    print("Retrieved {0} rows".format(len(data_set)))
    training_set, test_set = splitDataSet(data_set, 0.8)
    print("Size of training set is {0} and test_set is {1}".format(len(training_set), len(test_set)))
    classifier = Classifier()
    classifier.train(training_set)

    prediction_list = []

    for vector in test_set:
        prediction = classifier.predict(vector)
        prediction_list.append(prediction)

    print("Accuracy: ",nb.getAccuracy(prediction_list,test_set))

main()