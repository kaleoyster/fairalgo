"""
title: A fair online algorithm
description:
    An online algorithm that attempts to provide fair allocation of resources.
    This algorithm is also sensitive to concept drift
    The algorithm simulates streaming datasets, which is equal to stat
"""

__author__='Akshay Kale'
__copyright__='GPL'
__email__='akale@unomaha.edu'

import numpy
import pandas as pd
import random
import csv
import types
import matplotlib.pylab as plt
from collections import defaultdict
from sklego.metrics import equal_opportunity_score
from sklego.metrics import p_percent_score
from sklearn.linear_model import LogisticRegression

# TODO:
    # 1. Add fairness metrics
    # 2. Use test fairness code to build streaming pipeline simulation
def read_header(filename):
    """
    description:
    args:
    returns: 
    """
    with open(filename, 'r') as headerFile:
        headerReader = csv.reader(headerFile, delimiter=',')
        header = next(headerReader)
    return header

def return_header():
    """
    description: returns the header of adult dataset
    args:
        None
    returns:
        header (list): a list of column names of adult dataset
    """
    header = ['age',
              'workclass',
              'fnlwgt',
              'education',
              'education-num',
              'marital-status',
              'occupation',
              'relationship',
              'race',
              'sex',
              'capital-gain',
              'capital-loss',
              'hours-per-week',
              'native-country',
              'salary'
            ]
    return header

def read_csv(filename):
    """
    description: reads csv file and return list of values
    args:
        filename (string): path of the file
    returns:
        data (list of list): returns a list of list (column values)
    """
    data = list()
    with open(filename, 'r') as csvFile:
        csvReader = csv.reader(csvFile, delimiter=',')
        header = next(csvReader)
        for row in csvReader:
            data.append(row)
    return data

def predict_record(record):
    """
    TODO
    description:
    args:
    returns:
    """
    return record[-1]

def fairness_metric(predictions, groundTruth):
    """
    TODO
    description:
    args:
    returns:
    """
    fairnessMetric = None
    return fairnessMetric

def simulation(data):
    """
    TODO
    description:
        simulate the online prediction of streaming records
    args:
    returns:
    """
    groundTruth = list()
    predicted = list()
    # implement variable windows:
    # implement function to identify sensitive variables:
        # 1. age
        # 2. race
        # 3. native country 
        # 4. sex
    #windowOfData = select_window(data, groudTruth)
    #results = predict_record(windowOfData)
    #fairness = compute_fairness(results)
    for record in data:
        try:
            predicted.append(predict_record(record))
        except:
            print("Invalid record", record)

    return predicted

def create_window(data, columnNames, windowSize=10):
    """
    description:
        creates windows of dataset and returns a dictionary of attributes
    args:
        data (list of list): contains a list of list of dataset values
        columnNames (list): list of attribute names

    returns:
        dictionary (dict):
    """
    label = 'salary'
    sensitiveVariable='sex'

    senVarDict = {' Female': 1,
                  ' Male': 0 }

    labelDict = {' <=50K': 1,
                 ' >50K': 0 }

    data = data[:windowSize]
    dictionary = defaultdict(list)
    for row in data:
        for colName, val in zip(columnNames, row):
            if colName == sensitiveVariable:
                binaryVal =  senVarDict[val]
                dictionary[colName].append(binaryVal)
            elif colName == label:
                binaryVal = labelDict[val]
                dictionary[colName].append(binaryVal)
            else:
                dictionary[colName].append(val)
    return dictionary

def create_Xy(dictionary, label):
    """
    description:
        create a pandas dataframe, separates X and y
    args:
        dictionary (dict):
        label (string): 
    """
    df = pd.DataFrame(dictionary)
    X = df.drop(columns=label)
    y = df[label]

    return X, y

def main():
    """
    Driver function
    """
    filename = 'dataset/adult.data'
    headerfile = 'dataset/adult.names'

    data = read_csv(filename)
    label = 'salary'

    # Train a model on the initial model

    # Every dictionary is resembles a window
    columnNames = return_header()
    windowData = create_window(data, columnNames)

    # create training and testing data
    X, y = create_Xy(windowData, label)

    # Test model 
    model = types.SimpleNamespace()
    model.predict = lambda X: numpy.array([' <=50K', ' <=50K', ' <=50K', ' <=50K', ' <=50K', ' <=50K', ' >50K', ' >50K', ' >50K', ' >50K'])
    model.predict = lambda X: numpy.array([0, 1, 0, 1, 1, 1, 1, 0, 0, 0])

    # print equal opportunity score
    for time in range(100):
        print('equal opportunity score:', equal_opportunity_score(sensitive_column="sex")(model, X, y))


    #print(simulation(data))

if __name__ =='__main__':
    main()
