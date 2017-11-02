# imports
import numpy as np

# public symbols
# 'NaiveBayes1'だけ呼び出す!と宣言している(今回は特に効果なし)
__all__ = ['NaiveBayes1']


class NaiveBayes1(object):
    """
    Naive Bayes class (1)
    """
    def __init__(self):
        """
        Constructor
        データに依存しないパラメータは無いので引数はselfのみ
        事前分布、尤度のデータ数はデータに依存するので現段階ではNone
        """
        self.proY_ = None
        self.likeXY_ = None

    def fit(self, X, y):
        """
        Fitting model
        :param self: 
        :param X: 
        :param Y: 
        :return: 
        """

        #constants
        numSamples = X.shape[0]
        numFeatures = X.shape[1]
        numClasses = 2
        numFvalues = 2


        # check the size of y
        if numSamples != len(y):
            raise ValueError('Mismatched number of samples.')

        # count up n[yi=y]
        nY = np.zeros(numClasses, dtype = int)
        for i in range(numSamples):
            nY[y[i]] += 1

        # calc proY_
        self.proY_ = np.empty(numClasses, dtype = float)
        for i in range(numClasses):
            self.proY_[i] = nY[i] / numSamples

        # count up n[x_ij=xj, yi=y]
        nXY = np.zeros((numFeatures, numFvalues, numClasses), dtype = int)
        for i in range(numSamples):
            for j in range(numFeatures):
                nXY[j, X[i,j], y[i]] += 1

        # calc likeXY_
        self.likeXY_ = np.empty((numFeatures, numFvalues, numClasses), dtype = float)
        for j in range(numFeatures):
            for xi in range(numFvalues):
                for yi in range(numClasses):
                    self.likeXY_[j,xi,yi] = nXY[j,xi,yi] / float(nY[yi])


    def predict(self, X):
        """
        Predict model
        :param self: 
        :param X: 
        :return: 
        """

        # constants
        numSamples = X.shape[0]
        numFeatures = X.shape[1]

        # memory for return values
        y = np.empty(numSamples, dtype = int)

        #for each feature in X
        for i, xi in enumerate(X):

            # calc joint probability
            logpXY = (np.log(self.proY_)) + np.sum(np.log(self.likeXY_[np.arange(numFeatures),xi,:]), axis=0)

           #predict class
            y[i] = np.argmax(logpXY)

        return y