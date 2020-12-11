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
import random
import csv

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

def main():
    """
    Driver function
    """
    filename = 'dataset/adult.data'
    headerfile = 'dataset/adult.names'
    data = read_csv(filename)
    # read data using a pandas pd
    # define parameters

    print(simulation(data))

if __name__ =='__main__':
    main()
