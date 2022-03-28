# This is a sample Python script.

# Press Mayús+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import pandas as pd
import numpy as np

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    randomMatrix=np.random.rand(2,3)
    #1 - Dada una matriz en formato numpy array, donde cada fila de la matriz representa un vector matemático:
    #Computar las normas l0, l1, l2, l-infinito
    #    l0: número de elementos diferentes a cero en el vector
    def calculateNorms(randomMatrix):
        l0=np.count_nonzero(randomMatrix)
        l1=np.sum(np.absolute(randomMatrix),axis=1)
        l2=np.linalg.norm(randomMatrix,axis=1)
        lInf=randomMatrix.max(axis=1)
        return l0,l1,l2,lInf



    #Ej 2 ANDA
    def orderArraysByNorm(randomMatrix):
        lowToHighOrder = calculateNorms(randomMatrix)[2].argsort(kind='mergesort')
        highToLowOrder=np.flip(lowToHighOrder)
        finalMatrix=randomMatrix[highToLowOrder]
        return finalMatrix

    #Ej 3 ANDA
    users_identifiers = [15, 12, 14, 10, 1, 2, 1]
    users_index = [0, 1, 2, 3, 4, 5, 4]
    class IDX():
        def __init__(self,ids):
            self.ids=ids
        def get_users_id(self):
            matrix=-np.ones(np.max(self.ids)+1)
            ranges= np.arange(len(self.ids))
            matrix[self.ids]=ranges
            return matrix
        def get_users_idx(self):
            return self.ids
    ids=IDX(users_identifiers)
    print(ids.get_users_id()[2])


    #Ej 4 ANDA


    truth =         [1, 1, 0, 1, 1, 1, 0, 0, 0, 1]
    prediction =    [1, 1, 1, 1, 0, 0, 1, 1, 0, 0]

    def getMetrics(truth,prediction):
        sumed=np.add(truth,prediction)
        print(sumed)
        TP=np.floor_divide(sumed,2)
        TN=np.add(np.floor_divide(sumed,-2),np.ones(len(truth)))
        precision=np.divide(np.sum(TP),np.sum(prediction))
        recall=np.divide(np.sum(TP),np.sum(truth))
        accuracy=np.divide(np.sum(TP)+np.sum(TN), len(truth))
        return precision,recall,accuracy

    #EJ 5 ANDANDO
    q_id = [1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4]
    predicted_rank = [0, 1, 2, 3, 0, 1, 2, 0, 1, 2, 3, 4, 0, 1, 2, 3]
    truth_relevance = [True, False, True, False, True, True, True, False, False, False, False, False, True, False,
                       False, True]
    def getAverageQueryPrecision(q_id,predicted_rank,truth_relevance):
        correctGuesses=np.array(q_id)[truth_relevance]
        IDS,countOfEachItem=np.unique(np.array(q_id), return_counts=True)
        elementsInGuesses, countGuesses = np.unique(correctGuesses, return_counts=True)
        queryPrecisionArray=np.zeros(max(q_id))
        queryPrecisionArray[elementsInGuesses-1]=countGuesses
        AQP=queryPrecisionArray/countOfEachItem
        return np.sum(AQP)/len(AQP)

    #EJ 6
    def calcDistanceToCentroids(X,C):
        vectoresResultatntes = X - C[:, np.newaxis]
        normasResultatnes = np.transpose(np.linalg.norm(vectoresResultatntes, axis=2))
        return normasResultatnes
    X = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    C = [[1, 0, 0], [0, 1, 1]]
    X=np.array(X)
    C=np.array(C)
    print(calcDistanceToCentroids(X,C))

    #EJ 7
    def minDistToCentroid(distancesToCentroids):
        return np.argmin(distancesToCentroids, axis=1)
    def labelCluster(X,C):
        return minDistToCentroid(calcDistanceToCentroids(X,C))

    print(labelCluster(X,C))

    # EJ 8
    def KMeans(arrays,quantityOfClusters):
        iterations=150
        centroids=np.random.rand(quantityOfClusters,3)
        for j in range(iterations):
            distanceToCentroid = calcDistanceToCentroids(arrays, centroids)
            minimalDistToCentroids = minDistToCentroid(distanceToCentroid)
            for i in range(quantityOfClusters):
                centroids[i]=np.mean(arrays[minimalDistToCentroids==i],axis=0)
        return centroids

    #EJ 9

    class Metric:
        def __init__(self):
            print("init")

    class Normas(Metric):
        def __init__(self,matrix):
            self.matrix=matrix
        def __call__(self):
            return calculateNorms(self.matrix)
















# See PyCharm help at https://www.jetbrains.com/help/pycharm/
