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
import math


def mean(number_list):
    return sum(number_list) / len(number_list)


def standardDeviation(number_list):
    avg = mean(number_list)
    variance = sum([pow(x - avg, 2) for x in number_list]) / float(len(number_list) - 1)
    return math.sqrt(variance)


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
