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

    print("Accuracy: ",nb.getAccuracy(prediction_list,test_set))```
    
### Linear Regression

* Sample

  ```python
      input_matrix = [[1, float(x)] for x in open("DataSet/hieght_age/ex2x.dat").read().splitlines()]
      output = [[float(x)] for x in open("DataSet/hieght_age/ex2y.dat").read().splitlines()]
      
      h_theta = partial(cost_function,input_matrix,output)
      
      # creating regressor 
      linear = linear_regression.LinearRegression(input_matrix,output)
      # In case you want to use gradient decent to calculate the function use this:
      # linear = linear_regression.LinearRegression(input_matrix,output,True,0.07,[0,0])
      
      # To plot the points and the output of regression
      x = [x for (i, x) in linear.input]
      y_cal = []
      for i in linear.input:
          # The function getValueForInput(input_vector) will return the prediction of
          # what the output will be for the given input 
          y_cal.append(linear.getValueForInput(i))
          
      plt.scatter(x, linear.output, label="Points")
      plt.plot(x, y_cal, label="Regression")
      plt.legend()
      plt.xlabel("age")
      plt.ylabel("Height")
      plt.title("Regression modeling the height of boys with age")
      plt.show()
  ```
  [![Sample curve](https://s30.postimg.org/nje2smnch/figure_1.png)](https://postimg.org/image/pb71nj6p9/)
  
