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

from functools import partial

import numpy as np

from pydatalearn.calculus import gradientDecent
from pydatalearn.regression import cost_function


class LinearRegression:

    def __calculate_theta_normal_equation(self):
        inp = np.matrix(self.X)
        y = np.matrix(self.Y)
        theta = (inp.T * inp).I * inp.T * y  # The Normal Equation => (X'X)^-1 * X' * Y
        return theta.tolist()

    def __init__(self, input_matrix, output_matrix, user_gradient_decent=False,step_size=None,start_vector=None):
        """
        :param input_matrix:
        :param output_matrix:
        :param user_gradient_decent: Whether to use gradient decent to calculate parameters
        :param step_size:
        :param start_vector:
        """
        self.input = input_matrix
        self.output = output_matrix
        self.X = np.matrix(input_matrix)
        self.Y = np.matrix(output_matrix)
        if user_gradient_decent:
            h_theta = partial(cost_function, input_matrix, output_matrix)
            self.theta = gradientDecent(h_theta,start_vector,step_size)
        else:
            self.theta = self.__calculate_theta_normal_equation()
            self.theta = list(self.theta)
            self.theta = [i[0] for i in self.theta]

    def getValueForInput(self,row):
        val = 0
        for i in range(len(self.theta)):
            val += row[i]*self.theta[i]
        return val

    def getTheta(self):
        return self.theta
