# Ashwin Babu
# Machine Learning - KNN Algorithm implementation 


import csv
import random
import math
import operator

with open('gender.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        print(', '.join(row))
        
def loadFile(filename, split_ratio, trainingData=[], testData=[] ):
    with open(filename) as csvfile:
        rows = csv.reader(csvfile, delimiter=',')
        data = list(rows)
        for x in range(len(data)):
            for y in range(3):
                data[x][y] = int(data[x][y])
            if random.random() < split_ratio:
                trainingData.append(data[x])
            else:
                testData.append(data[x])
                
                
# Cartesian distance calculator function for KNN algorithm
def euclideanDistance(item1, item2, length):
    cal_dist = 0
    for x in range(length):
        cal_dist += pow((item1[x] - item2[x]), 2)
    return math.sqrt(cal_dist)

# Function to check the nearest points from each test data to predict the class
def checkNeighbors(trainingData, test, k):
    distance_measure =[]
    length = len(test)-1
    for x in range(len(trainingData)):
        dist = euclideanDistance(test, trainingData[x], length)
        distance_measure.append(( dist, trainingData[x]))
    distance_measure.sort(key=operator.itemgetter(0))
    #print(distance_measure)
    neighbors =[]
    for x in range(k):
        neighbors.append(distance_measure[x][1])
    return neighbors


# Function to see the majority class near the point and predict the class
def determineClass(neighbors):
    classMajority = {}
    for x in range(len(neighbors)):
        classification = neighbors[x][-1]
        if classification in classMajority:
            classMajority[classification] +=1
        else:
            classMajority[classification] = 1
    sortedMajority = max(classMajority.items(), key=operator.itemgetter(1))[0]
    return sortedMajority[0][0]


# Function to calculate the accuracy of prediction
def accuracy(test, prediction):
    identified=0
    for x in range(len(test)):
        if test[x][-1] is prediction[x]:
            identified +=1
    return (identified/float(len(test))) * 100.0


def main():
    # Data from file
    trainingData = []
    testData = []
    # Splitting the datta 67/33 which is an universal value to check accuracy
    split = 0.67
    loadFile('gender.csv', split, trainingData, testData)
    print('Training Data: ' + repr(len(trainingData)))
    print('Test Data: ' + repr(len(testData)))
    # Predicting the class
    predictions = []
    k=input('Enter value for k: ')
    k = int(k)
    for x in range(len(testData)):
        neighbors = checkNeighbors(trainingData, testData[x], k)
        result = determineClass(neighbors)
        predictions.append(result)
        print('--> predicted=' + repr(result) + '-->actual=' + repr(testData[x][-1]))
    accuracy_1 = accuracy(testData, predictions)
    print('Accuracy:' + repr(accuracy_1) + '%')
    
main()