
import numpy as np


randomMatrix=np.random.rand(2,3)
#1 - Dada una matriz en formato numpy array, donde cada fila de la matriz representa un vector matemático:
#Computar las normas l0, l1, l2, l-infinito
#    l0: número de elementos diferentes a cero en el vector
def calculateNorms(randomMatrix):
    l0=np.count_nonzero(randomMatrix,axis=1)
    l1=np.sum(np.absolute(randomMatrix),axis=1)
    l2=np.linalg.norm(randomMatrix,axis=1)
    lInf=randomMatrix.max(axis=1)
    return dict(l0=l0,l1=l1,l2=l2,lInf=lInf)



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




#Ej 4 ANDA
truth =         [1, 1, 0, 1, 1, 1, 0, 0, 0, 1]
prediction =    [1, 1, 1, 1, 0, 0, 1, 1, 0, 0]

def getMetrics(truth,prediction):
    sumed=np.add(truth,prediction)
    TP=np.floor_divide(sumed,2)
    TN=np.add(np.floor_divide(sumed,-2),np.ones(len(truth)))
    precision=np.divide(np.sum(TP),np.sum(prediction))
    recall=np.divide(np.sum(TP),np.sum(truth))
    accuracy=np.divide(np.sum(TP)+np.sum(TN), len(truth))
    return dict(precision=precision,recall=recall,accuracy=accuracy)

#EJ 5 ANDANDO
q_id = [1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4]
predicted_rank = [0, 1, 2, 3, 0, 1, 2, 0, 1, 2, 3, 4, 0, 1, 2, 3]
truth_relevance = [True, False, True, False, True, True, True, False, False, False, False, False, True, False,
                       False, True]
def getAverageQueryPrecision(q_id,predicted_rank,truth_relevance):
    correctGuesses=np.array(q_id)[truth_relevance]
    IDS,countOfEachItem=np.unique(np.array(q_id), return_counts=True)
    elementsInGuesses, countGuesses = np.unique(correctGuesses, return_counts=True)
    queryPrecisionArray=np.zeros(max(q_id)-min(q_id)+1)
    queryPrecisionArray[elementsInGuesses-min(elementsInGuesses)]=countGuesses
    AQP=queryPrecisionArray/countOfEachItem
    return np.sum(AQP)/len(AQP)


#EJ 6
def calcDistanceToCentroids(X,C):
    vectoresResultatntes = X - C[:, np.newaxis]
    normasResultatnes = np.transpose(np.linalg.norm(vectoresResultatntes, axis=2))
    return normasResultatnes


#EJ 7
def minDistToCentroid(distancesToCentroids):
    return np.argmin(distancesToCentroids, axis=1)
def labelCluster(X,C):
    return minDistToCentroid(calcDistanceToCentroids(X,C))



# EJ 8
def KMeans(arrays,quantityOfClusters):
    iterations=150
    centroids=np.random.rand(quantityOfClusters,3)
    for j in range(iterations):
        distanceToCentroid = calcDistanceToCentroids(arrays, centroids)
        minimalDistToCentroids = minDistToCentroid(distanceToCentroid)
        for i in range(quantityOfClusters):
            centroids[i]=np.mean(arrays[minimalDistToCentroids==i],axis=0)
    return centroids,minimalDistToCentroids

    #EJ 9

class AbstractMetric:
    def __init__(self, **kwargs):
        self.parameters = kwargs

    def __call__(self, *args, **kwargs):
        pass


class Precision(AbstractMetric):
    def __init__(self,**kwargs):
        AbstractMetric.__init__(self,**kwargs)
    def __call__(self):
        return getMetrics(self.parameters["truth"],self.parameters["prediction"])["precision"]

class Recall(AbstractMetric):
    def __init__(self,**kwargs):
        AbstractMetric.__init__(self,**kwargs)
    def __call__(self):
        return getMetrics(self.parameters["truth"],self.parameters["prediction"])["recall"]

class Accuracy(AbstractMetric):
    def __init__(self,**kwargs):
        AbstractMetric.__init__(self,**kwargs)
    def __call__(self):
        return getMetrics(self.parameters["truth"],self.parameters["prediction"])["accuracy"]

class AQP(AbstractMetric):
    def __init__(self,**kwargs):
        AbstractMetric.__init__(self,**kwargs)
    def __call__(self):
        return getAverageQueryPrecision(self.parameters["truth"],self.parameters["prediction"],self.parameters["truth_relevance"])

class ComputeMetrics():
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.metrics = {}

    def computeAllMetrics(self):
        myMetrics = [AQP,Precision,Recall,Accuracy]
        for metric in myMetrics:
            currentMetric = metric(**self.kwargs)
            self.metrics[metric.__name__] = currentMetric()
        return self.metrics
















# See PyCharm help at https://www.jetbrains.com/help/pycharm/
