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

# develope a basic linear regression model on the adult dataset
def detect_concept_dift(currentStream, previousStream):
    threshold = 0
    difference = 0
    for currentInstance, previousAvg in zip(currentStream, previousStream):
        difference = difference + (previousAvg - currentInstance)

    if difference > threshold:
        return True
    else:
        return False

def online_algorithm(stream):
    classification = list()
    for instance in stream:
        classification.append(instance[-1])
    return classification

def produce_stream(randomGenerator):
    noOfAttributes = 3
    streamLength = 10

    gender = ['Male', 'Female']
    race = ['White', 'Black']
    label = ['No', 'Yes']

    streamAge = randomGenerator.uniform(15, 67, streamLength)
    streamGender = random.choices(gender, k=streamLength)
    streamRace = random.choices(race, k=streamLength)
    streamLabel = random.choices(label, k=streamLength)

    streamOfInstances = list()
    for time in range(0, streamLength):
        instance = list()
        instance.append(streamAge[time])
        instance.append(streamGender[time])
        instance.append(streamRace[time])
        instance.append(streamLabel[time])
        streamOfInstances.append(instance)
    return streamOfInstances

def main():
    """
    Driver function
    """
    randomGenerator = numpy.random.RandomState()

    for i in range(0, 10):
        print("stream:", i)
        stream = produce_stream(randomGenerator)
        classificationResults  = online_algorithm(stream)
        #classificationResults  = online_algorithm(stream, fairnessValue)
        #fairnessValue = evalulate_fairness(classificationResults)
        #print(results)

if __name__ =='__main__':
    main()
