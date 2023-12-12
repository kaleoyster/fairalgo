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
from collections import Counter
from tqdm import tqdm
from collections import defaultdict
from sklego.metrics import equal_opportunity_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.tree import DecisionTreeClassifier
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
        This function also preprocess dataset
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
    #print(dictionary)
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

def create_col_value_mapping(colList):
    mappings = {item:index for index, item in enumerate(colList)}
    return mappings

def preprocess_columns(X, y):
    """
    description:
    args:
    returns
    """
    workclassMap = create_col_value_mapping(X['workclass'].unique())
    X['workclass'] = X['workclass'].map(workclassMap)

    educationMap = create_col_value_mapping(X['education'].unique())
    X['education'] = X['education'].map(educationMap)

    maritalstatusMap = create_col_value_mapping(X['marital-status'].unique())
    X['marital-status'] = X['marital-status'].map(maritalstatusMap)

    occupationMap = create_col_value_mapping(X['occupation'].unique())
    X['occupation'] = X['occupation'].map(occupationMap)

    relationshipMap = create_col_value_mapping(X['relationship'].unique())
    X['relationship'] = X['relationship'].map(relationshipMap)

    raceMap = create_col_value_mapping(X['race'].unique())
    X['race'] = X['race'].map(raceMap)

    sexMap = create_col_value_mapping(X['sex'].unique())
    X['sex'] = X['sex'].map(sexMap)

    countryMap = create_col_value_mapping(X['native-country'].unique())
    X['native-country'] = X['native-country'].map(countryMap)

    return X, y

def create_model(X, y):
    model = DecisionTreeClassifier()
    model = model.fit(X, y)
    return model

def compute_sensitive_ratio(windowData):
    """
    Description:
    args:
    returns:
    """
    maleCount = Counter(windowData)[0]
    femaleCount = Counter(windowData)[1]
    ratio = maleCount / (femaleCount + maleCount)
    return ratio


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
    sensitiveVariableCount = list()
    for time in tqdm(range(numOfSimulation)):
       windowData = create_window(data, columnNames)
       X, y = create_Xy(windowData, label)
       sensitiveVarCount = compute_sensitive_ratio(X['sex'])
       X, y = preprocess_columns(X, y)
       eqTemp = equal_opportunity_score(sensitive_column="sex")(model, X, y)
       equalOpportunities.append(eqTemp)
       sensitiveVariableCount.append(sensitiveVarCount)
    return equalOpportunities, sensitiveVariableCount

def plot_equal_opportunity(equalOpData):
    """
    description:

    args:
    returns:
    """
    plt.plot(equalOpData)
    plt.title('Equal opportunity score v. runs')
    plt.xlabel('Number of runs')
    plt.ylabel('Equal opportunity score')
    plt.savefig('equalOpportunityScore.png')
    #plt.show()

def plot_sensitive_count(sensitiveCount):
    """
    description:
    args:
    returns:
    """
    plt.plot(sensitiveCount)
    plt.title('Equal opportunity score v. runs')
    plt.xlabel('Number of runs')
    plt.ylabel('Equal opportunity score')
    plt.savefig("sensitiveCount.png")
    #plt.show()

def main():
    """
    Driver function
    """
    # files
    filename = 'dataset/adult.data'
    headerfile = 'dataset/adult.names'

    # data
    trainFilename = 'dataset/trainadult.data'
    trainFilename = 'dataset/trainadult.names'

    data = read_csv(filename)
    label = 'salary'

    # Every dictionary is resembles a window
    columnNames = return_header()

    # Create a window -> add an aspect of time
    windowData = create_window(data, columnNames, windowSize=1000)
    X, y = create_Xy(windowData, label)

    # Train a model on the initial model
    X, y = preprocess_columns(X, y)
    model = create_model(X, y)

    # Visualization 
    equalOpData, sensitiveCount = simulate_eo(data, columnNames, label, model)
    plot_equal_opportunity(equalOpData)
    plot_sensitive_count(sensitiveCount)

if __name__ =='__main__':
    main()
