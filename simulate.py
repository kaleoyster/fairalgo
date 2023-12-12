"""
title: A fair online algorithm
description:
    An online algorithm that attempts to provide fair allocation of resources.
    This algorithm is also sensitive to concept drift
    The algorithm simulates streaming datasets, which is equal to stat...?
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
from tqdm import tqdm
from collections import defaultdict
from sklego.metrics import equal_opportunity_score
from sklego.metrics import p_percent_score
from sklearn.linear_model import LogisticRegression

def read_header(filename):
    """
    *Dead code*
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
    *Dead Code*
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

def get_data(data, windowSize):
    """
    """
    windowData = list()
    currentWindowSize = 0
    while  currentWindowSize < windowSize:
        tempData = data.pop()
        if tempData != list():
            windowData.append(tempData)
            currentWindowSize = currentWindowSize + 1
    return windowData

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

    windowData = get_data(data, windowSize)

    # prepare_data
    dictionary = defaultdict(list)
    for row in windowData:
        for colName, val in zip(columnNames, row):
            if colName == sensitiveVariable:
                binaryVal =  senVarDict.get(val)
                if binaryVal == None:
                   binaryVal = 0
                dictionary[colName].append(binaryVal)
            elif colName == label:
                binaryVal = labelDict.get(val)
                if binaryVal == None:
                   binaryVal = 0
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

    returns:
        X (list of list): list of independent variables
        y (list): classification label / ground truth
    """
    df = pd.DataFrame(dictionary)
    X = df.drop(columns=label)
    y = df[label]
    return X, y

def simulate_eo(data, columnNames, label, model, numOfSimulation=100):
    """
    description:
        runs simulation based on the number of simulations

    args:
        data (list of list): list of attributes and values
        label (string): classification label / ground truth
        model (classification model): sckit classification model
        numOfSimulation (int): number of simulations to run

    returns:
        equalOpportunities (list): list of equal opportunities
    """
    numOfSimulation = 1000
    equalOpportunities = list()
    for time in tqdm(range(numOfSimulation)):
       windowData = create_window(data, columnNames)
       X, y = create_Xy(windowData, label)
       eqTemp = equal_opportunity_score(sensitive_column="sex")(model, X, y)
       equalOpportunities.append(eqTemp)
    return equalOpportunities

def plot_equal_opportunity(equalOpData):
    plt.plot(equalOpData)
    plt.title('Equal opportunity score v. runs')
    plt.xlabel('Number of runs')
    plt.ylabel('Equal opportunity score')
    plt.show()

def main():
    """
    Driver function
    """
    filename = 'dataset/adult.data'
    headerfile = 'dataset/adult.names'

    data = read_csv(filename)
    label = 'salary'

    # Every dictionary is resembles a window
    columnNames = return_header()

    # Train a model on the initial model (TODO)
    # define model
    model = types.SimpleNamespace()
    model.predict = lambda X: numpy.array([0, 1, 0, 1, 1, 1, 1, 0, 0, 0])

    # print equal opportunity score
    equalOpData = simulate_eo(data, columnNames, label, model)
    plot_equal_opportunity(equalOpData)

if __name__ =='__main__':
    main()
