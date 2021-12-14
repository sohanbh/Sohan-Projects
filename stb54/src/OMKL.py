import argparse
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import math


def accuracy(y, y_hat) -> float:
    """
    Evaluate the accuracy of a set of predictions.

    :param y: Labels (true data)
    :param y_hat: Predictions
    :return: Accuracy of predictions
    """
    correct = 0
    wrong = 0
    for (real, prediction) in zip(y, y_hat):
        if real == prediction:
            correct += 1
        else:
            wrong += 1
    return (correct) / (correct + wrong)

def expKernel(x, y):
    #Exponential Kernel
    diff = np.array(x) - np.array(y)
    function = math.exp(-(np.linalg.norm(diff)) * 10)
    return function

def gaussianKernel(x, y):
    #Gaussian Kernel
    diff = np.array(x) - np.array(y)
    function = math.exp(-(np.linalg.norm(diff) * np.linalg.norm(diff)) * 10)
    return function

def inverseKernel(x, y):
    #Inverse MultiQuadratic Kernel
    diff = np.array(x) - np.array(y)
    if np.linalg.norm(diff) == 0:
        function = 1
    else:
        function = 1/math.sqrt(np.linalg.norm(diff) * np.linalg.norm(diff))
    return function

def cauchyKernel(x, y):
    #Cauchy Kernel
    diff = np.array(x) - np.array(y)
    if np.linalg.norm(diff) == 0:
        function = 1
    else:
        function = 1/(1 + ((np.linalg.norm(diff) * np.linalg.norm(diff))))
    return function


def predictKernel(kernel, x, features, labels, bins, binList):
    """
    Helper function used to get the classification of and example on a kernel classifier
    :param kernel: String, states which kernel to use
    :param x: the example kernel is predicting
    :param features: the dataset of features
    :param labels: the labels of the features
    :param bins: boolean, whether or not to use bins (research extension)
    :param binList: the list of bins (research extension)
    :return: return the prediction of the kernel classifier
    """
    kernels = []
    for i in range(0, len(features.index)):
        if not bins:
            y = features.iloc[i].to_numpy()
        else:
            y = binArray(features.iloc[i].to_numpy(), binList)
        if kernel == "gaussian":
            kernels.append(gaussianKernel(x, y))
        elif kernel == "cauchy":
            kernels.append(cauchyKernel(x, y))
        elif kernel == "exp":
            kernels.append(expKernel(x, y))
        elif kernel == "inverse":
            kernels.append(inverseKernel(x, y))
    weights = []
    for kernel in kernels:
        weights.append(len(features.index) * kernel / np.sum(kernels))
    p = np.dot(weights, labels) / len(features.index)
    return p

def binArray(array, binList):
    """
    Helper method used for research extension
    :param array: the array of values of a specific example
    :param binList: a dictionary of bins
    :return: the new array of binned features
    """
    example = array
    for i in range(0, len(example)):
        if i in binList.keys():
            example[i] = np.digitize(example[i], binList[i])
    return example



class KernelClassifier:
    def __init__(self, features, labels, bins):
        self.features = features
        self.bins = bins
        self.labels =  np.array(labels)
        for i in range(0, len(self.labels)):
            if self.labels[i] == 0:
                self.labels[i] = -1
        self.weights = None
        self.kAdd = None
        self.binDict = None

    def train(self):
        kAdd = [0, 0, 0, 0]
        weights = [1, 1, 1, 1]
        features = self.features
        labels = self.labels
        binDict = {}
        if self.bins:
            x = 0
            for column in features:
                if len(list(set(features[column]))) > 2:
                    binDict[x] = np.arange(features[column].min(), features[column].max(), (features[column].max() - features[column].min())/10)
                x += 1
            self.binDict = binDict
        for i in range(0, len(features.index)):
            if self.bins:
                example = binArray(features.iloc[i].to_numpy(), binDict)
            else:
                example = features.iloc[i].to_numpy()
            kernelPredict = [predictKernel("gaussian", example, features, labels, self.bins, binDict) + kAdd[0], predictKernel("cauchy", example, features, labels, self.bins, binDict) + kAdd[1],
                           predictKernel("exp", example, features, labels, self.bins, binDict) + kAdd[2], predictKernel("inverse", example, features, labels, self.bins, binDict) + kAdd[3]]
            binaryPredictions = []
            for k in range(0, len(kernelPredict)):
                if kernelPredict[k] >= 0:
                    binaryPredictions.append(1)
                else:
                    binaryPredictions.append(-1)
            label = labels[i]
            for j in range(0, len(weights)):
                if (label * binaryPredictions[j]) <= 0:
                    z = 1
                else:
                    z = 0
                if label == -1:
                    kAdd[j] -= z*label*kernelPredict[j]
                else:
                    kAdd[j] += z*label*kernelPredict[j]
                weights[j] *= 0.8**z
        self.kAdd = kAdd
        self.weights = weights

    def predict(self, dataset, labels):
        kAdd = self.kAdd
        weights = self.weights
        bins = self.bins
        binDict = self.binDict
        normWeights = weights / np.sum(weights)
        features = dataset
        trainingFeatures = self.features
        trainingLabels = self.labels
        labels = labels.to_numpy()
        for i in range(0, len(labels)):
            if labels[i] == 0:
                labels[i] = -1
        predictions = []
        for i in range(0, len(features.index)):
            if bins:
                example = binArray(features.iloc[i].to_numpy(), binDict)
            else:
                example = features.iloc[i].to_numpy()
            kernelPredict = [predictKernel("gaussian", example, trainingFeatures, trainingLabels, bins, binDict) + kAdd[0],
                             predictKernel("cauchy", example, trainingFeatures, trainingLabels, bins, binDict) + kAdd[1],
                             predictKernel("exp", example, trainingFeatures, trainingLabels, bins, binDict) + kAdd[2],
                             predictKernel("inverse", example, trainingFeatures, trainingLabels, bins, binDict) + kAdd[3]]
            binaryPredictions = []
            for i in range(0, len(kernelPredict)):
                if kernelPredict[i] >= 0:
                    binaryPredictions.append(1)
                else:
                    binaryPredictions.append(-1)
            predictions.append(np.dot(normWeights, binaryPredictions))
            for i in range(0, len(predictions)):
                if predictions[i] > 0:
                    predictions[i] = 1
                else:
                    predictions[i] = -1
        print('Accuracy: ', accuracy(labels, predictions))



def projectClassifier(data_path, bins):
    data = data_path
    df = pd.read_csv(data, index_col=0)
    sorted = df.sort_values('label', ascending=True)
    s1 = sorted.head(500)
    s2 = sorted.tail(500)
    trainingDataframe = pd.concat([s1, s2], axis = 0).sample(frac=1, random_state=123142141)
    trainingFeatures = trainingDataframe.drop('label', axis = 1)
    trainingLabels = trainingDataframe['label']
    predictDataframe = pd.read_csv(data, index_col = 0).sample(frac=1, random_state=12314).tail(1000)
    predictFeatures = predictDataframe.drop('label', axis = 1)
    predictLabels = predictDataframe['label']

    c = KernelClassifier(trainingFeatures, trainingLabels, bins)
    c.train()
    c.predict(predictFeatures, predictLabels)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run a kernel classification algorithm')
    parser.add_argument('path', metavar='PATH', type=str, help='The path to the csv file of the data.')
    parser.add_argument('--use-bins', dest='bins', action = 'store_true',
                        help='separate continuous variables into bins')
    parser.set_defaults(bins=False)
    args = parser.parse_args()
    path = args.path
    bins = args.bins
    projectClassifier(path, bins)