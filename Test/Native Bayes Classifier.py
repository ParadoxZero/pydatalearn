"""
Dataset description:
https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.names
"""
from pydatalearn import naive_bayes_classifier as nv


def main():
    print("Loading pima-indians-diabetes.data.csv...")
    data_set = (nv.loadCSV("DataSet/pima-indians-diabetes.data.csv"))
    print("Retrieved {0} rows".format(len(data_set)))

    training_set, test_set = nv.splitDataSet(data_set, 0.67)

    print("Size of training set is {0} and test_set is {1}".format(len(training_set), len(test_set)))

    summaries = nv.summarizeByClass(training_set)

    prediction = []
    for vector in test_set:
        prediction.append(nv.predict(summaries,vector))

    print("Accuracy: ",nv.getAccuracy(prediction,test_set))

main()