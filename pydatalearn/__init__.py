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
import csv
import math
import random


def mean(number_list):
    return sum(number_list) / len(number_list)


def standardDeviation(number_list):
    avg = mean(number_list)
    variance = sum([pow(x - avg, 2) for x in number_list]) / float(len(number_list) - 1)
    return math.sqrt(variance)



def loadCSV(filename):
    lines = csv.reader(open(filename, 'r'))
    data_set = []
    for line in lines:
        data_set.append([float(x) for x in line])
    return data_set


def splitDataSet(data_set, ratio):
    data_set = list(data_set)
    train_size = int(len(data_set) * ratio)
    training_set = []
    for _ in range(train_size):
        training_set.append(data_set.pop(data_set.index(random.choice(data_set))))
    return training_set, data_set


