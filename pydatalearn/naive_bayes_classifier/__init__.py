import csv
import random

import math


def load_CSV(filename):
    lines = csv.reader(open(filename, 'r'))
    data_set = []
    for line in lines:
        data_set.append([float(x) for x in line])
    return data_set


def split_dataset(data_set, ratio):
    data_set = list(data_set)
    train_size = int(len(data_set) * ratio)
    training_set = []
    for _ in range(train_size):
        training_set.append(data_set.pop(data_set.index(random.choice(data_set))))
    return training_set, data_set


def separate_by_class(data_set):
    separated = {}
    for vector in data_set:
        if vector[-1] in separated:
            separated[vector[-1]].append(vector)
        else:
            separated[vector[-1]] = [vector, ]
    return separated


def mean(number_list):
    return sum(number_list) / len(number_list)


def standard_deviation(number_list):
    avg = mean(number_list)
    variance = sum([pow(x - avg, 2) for x in number_list]) / float(len(number_list) - 1)
    return math.sqrt(variance)


def summarize(dataset):
    summaries = [(mean(attribute), standard_deviation(attribute)) for attribute in zip(*dataset)]
    summaries.pop(-1)
    return summaries


def summarize_by_class(data_set):
    separated = separate_by_class(data_set)
    separate_summary = {}
    for classValue, instances in separated.items():
        separate_summary[classValue] = summarize(instances)
    return separate_summary


def calculateProbability(x, mean, stdev):
    exponent = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stdev, 2))))
    return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent

def calculateClassProbablity(summaries, input_vector):
    probablity = {}
    for key,classSummary in summaries.items():
        probablity[key] = 1
        for index,(mean,std) in enumerate(classSummary):
            probablity[key] *= calculateProbability(input_vector[index],mean,std)
    return probablity

def predict(summaries,vector):
    probability = calculateClassProbablity(summaries, vector)
    class_ = None
    best_prob = 0
    for key, value in probability.items():
        if value > best_prob:
            class_ = key
            best_prob = value
    return class_

def get_accuracy(prediction,test_set):
    correct_prediction = 0
    for i in range(len(test_set)):
        if prediction[i] == test_set[i][-1]:
            correct_prediction += 1
    return correct_prediction/len(test_set) * 100