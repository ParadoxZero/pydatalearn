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

def differentiate(func, h=0.00001):
    """
    Error prone after 2nd degree
    :param func: function to differentiate
    :param h: @type float, step size
    :return: function
    """

    def derivative(x):
        return (func(x + h) - func(x)) / h

    return derivative


def partialDifferentiate(func, variable_index, h=0.00001):
    def derivative(*input_values):
        values_with_h = [x + h if i == variable_index else x for i, x in enumerate(input_values)]
        return (func(values_with_h) - func(input_values)) / h

    return derivative


def gradientDecent(func, theta, step):
    """
    :param func: The function who's minima you want to find
    :param theta:
    :param step:
    :return: a list containing input corresponding to minimum value of function
    """
    D_func = []
    for i in range(len(theta)):
        D_func.append(partialDifferentiate(func, i))
    theta_old = list(theta)
    print(theta)
    is_cont = True
    i = 0
    while (is_cont):
        for index, value in enumerate(theta_old):
            theta[index] = theta_old[index] - step * D_func[index](*theta)
        print(i, ":", theta)
        i += 1
        change = 0
        for k in range(len(theta)):
            change = abs(theta[k] - theta_old[k])
        if change == 0:
            is_cont = False
        theta_old = list(theta)
    return theta
