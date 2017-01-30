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
import random
import math

from pydatalearn import mean, standardDeviation


def separateByClass(data_set):
    separated = {}
    for vector in data_set:
        if vector[-1] in separated:
            separated[vector[-1]].append(vector)
        else:
            separated[vector[-1]] = [vector, ]
    return separated


def summarize(dataset):
    summaries = [(mean(attribute), standardDeviation(attribute)) for attribute in zip(*dataset)]
    summaries.pop(-1)
    return summaries


def summarizeByClass(data_set):
    separated = separateByClass(data_set)
    separate_summary = {}
    for classValue, instances in separated.items():
        separate_summary[classValue] = summarize(instances)
    return separate_summary


def calculateProbability(x, mean, stdev):
    exponent = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stdev, 2))))
    return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent


def calculateClassProbablity(summaries, input_vector):
    probability = {}
    for key, classSummary in summaries.items():
        probability[key] = 1
        for index, (mean, std) in enumerate(classSummary):
            probability[key] *= calculateProbability(input_vector[index], mean, std)
    return probability


def predict(summaries, vector):
    probability = calculateClassProbablity(summaries, vector)
    class_ = None
    best_prob = 0
    for key, value in probability.items():
        if value > best_prob:
            class_ = key
            best_prob = value
    return class_


def getAccuracy(prediction, test_set):
    correct_prediction = 0
    for i in range(len(test_set)):
        if prediction[i] == test_set[i][-1]:
            correct_prediction += 1
    return correct_prediction / len(test_set) * 100
