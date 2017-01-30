"""
MIT License

Copyright (c) 2017 Sidhin S Thomas

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np


def cost_function(X, Y, *theta):
    """
    Find out the difference between the values from parameter and actual
    :param X: Input matrix
    :param Y: Actual output Matrix
    :param theta: Fitted parameter that's needed to be tested
    :return: Cost/error @float
    """
    X = np.matrix(X)
    Y = np.matrix(Y)
    theta = np.matrix(theta).transpose()
    prediction = X * theta
    # print(prediction)
    error = 0
    for i in range(len(prediction)):
        error += (Y.item(i, 0) - prediction.item(i, 0)) ** 2
    error = error / (2 * len(Y))
    # print("Cost: ",error)
    return error
