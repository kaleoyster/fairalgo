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

def read_header(filename):
    """
    description:
    args:
    returnsG
    """
    with open(filename, 'r') as headerFile:
        headerReader = csv.reader(headerFile, delimiter=',')
        header = next(headerReader)
    return header

def return_header():
    """
    description:
    args:
    returns:
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
    description:
    args:
    returns:
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
    description:
    args:
    returns:
    """
    return record[-1]

def fairness_metric(predictions, groundTruth):
    """
    description:
    args:
    returns:
    """
    fairnessMetric = None
    return fairnessMetric

def simulation(data):
    """
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
    print(simulation(data))

if __name__ =='__main__':
    main()
