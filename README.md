# pydatalearn [![PyPI version](https://badge.fury.io/py/pydatalearn.svg)](https://badge.fury.io/py/pydatalearn)

This is a library containing different data science models which can be used in machine learning applications like classification etc.

## Installing

```bash
  $ pip install pydatalearn
```

## How to use ?

### Naive Bayes Classifier

The data is assumed to contain the classification/label/class as the last **number** in `csv` file

* Example usage

  ```python
    from pydatalearn import naive_bayes_classifier as nb
    from pydatalearn.naive_bayes_classifier.classifie
    
    # loading the data from csv file
    data_set = (nb.loadCSV("DataSet/pima-indians-diabetes.data.csv"))
    
    #Split the data into training set and test set, if you wish to check the result (Optional)
    training_set, test_set = nb.splitDataSet(data_set, 0.8)
    
    classifier = Classifier()
    classifier.train(training_set)

    prediction_list = []

    for vector in test_set:
    
        # predict the class of the data
        prediction = classifier.predict(vector)
        
        prediction_list.append(prediction)

    print("Accuracy: ",nb.getAccuracy(prediction_list,test_set))
   
 ```
