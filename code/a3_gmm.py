from sklearn.model_selection import train_test_split
import numpy as np
import os, fnmatch
import random
from scipy.special import logsumexp

# dataDir = '/u/cs401/A3/data/'
dataDir = "../data/"


class theta:
    def __init__(self, name, M=8,d=13):
        self.name = name
        self.omega = np.zeros((M,1))
        self.mu = np.zeros((M,d))
        self.Sigma = np.zeros((M,d))


def log_b_m_x( m, x, myTheta, preComputedForM=[]):
    ''' Returns the log probability of d-dimensional vector x using only component m of model myTheta
        See equation 1 of the handout

        As you'll see in tutorial, for efficiency, you can precompute something for 'm' that applies to all x outside of this function.
        If you do this, you pass that precomputed component in preComputedForM

    '''
    M, d = myTheta.Sigma.shape
    top = -1/2 * np.sum(np.square(x-myTheta.mu[m])/myTheta.Sigma[m])
    pi = d/2*np.log(2*np.pi)
    root = 1/2*np.sum(np.log(myTheta.Sigma[m]))

    return top - pi - root

    
def log_p_m_x( m, x, myTheta):
    ''' Returns the log probability of the m^{th} component given d-dimensional vector x, and model myTheta
        See equation 2 of handout
    '''
    M, d = myTheta.Sigma.shape
    logB = np.array([log_b_m_x(i, x, myTheta) for i in range(M)])
    top = logB[m] + np.log(myTheta.omega[m, 0])
    bot = logsumexp(logB + np.log(myTheta.omega))
    return top - bot


def logLik( log_Bs, myTheta ):
    ''' Return the log likelihood of 'X' using model 'myTheta' and precomputed MxT matrix, 'log_Bs', of log_b_m_x

        X can be training data, when used in train( ... ), and
        X can be testing data, when used in test( ... ).

        We don't actually pass X directly to the function because we instead pass:

        log_Bs(m,t) is the log probability of vector x_t in component m, which is computed and stored outside of this function for efficiency. 

        See equation 3 of the handout
    '''
    return np.sum(logsumexp(log_Bs + np.log(myTheta.omega), axis=0))


def log_Bs(X, myTheta):

    M, d = myTheta.Sigma.shape
    T = X.shape[0]

    log_Bs = [- 1/2 * np.sum(np.square(X - myTheta.mu[m]) / myTheta.Sigma[m], axis=1) 
                - (d / 2 * np.log(2 * np.pi) 
                    + 1. / 2 * np.sum(np.log(myTheta.Sigma[m])))
                        for m in range(M)]
    
    return log_Bs


def train( speaker, X, M=8, epsilon=0.0, maxIter=20 ):
    ''' Train a model for the given speaker. Returns the theta (omega, mu, sigma)'''

    myTheta = theta( speaker, M, X.shape[1] )
    T, d = X.shape
    np.random.seed(0)
    indList = np.random.choice(T, M, replace=False)
    myTheta.mu = X[indList]
    myTheta.Sigma[:,:] = 1
    myTheta.omega[:,0] = 1/M

    previousL = float("-inf")
    diff = float("inf")
    i = 0

    while (i <= maxIter and diff > epsilon):
        # calculate log Bs to get current log likelihood
        log_bs = log_Bs(X, myTheta)
        curL = logLik(log_bs, myTheta)
        diff = abs(curL - previousL)
        logTop = log_bs + np.log(myTheta.omega)
        log_Ps = logTop - logsumexp(logTop, axis=0)
        i += 1
        previousL = curL
        # update
        for m in range(M):
            prob = np.exp(log_Ps[m])
            probSum = np.sum(prob)
            myTheta.omega[m] = probSum/T
            myTheta.mu[m] = np.dot(prob, X) / probSum
            myTheta.Sigma[m] = (np.dot(prob, np.square(X))/probSum) - (myTheta.mu[m]**2)

    return myTheta


def test( mfcc, correctID, models, k=5 ):
    ''' Computes the likelihood of 'mfcc' in each model in 'models', where the correct model is 'correctID'
        If k>0, print to stdout the actual speaker and the k best likelihoods in this format:
               [ACTUAL_ID]
               [SNAME1] [LOGLIK1]
               [SNAME2] [LOGLIK2]
               ...
               [SNAMEK] [LOGLIKK] 

        e.g.,
               S-5A -9.21034037197
        the format of the log likelihood (number of decimal places, or exponent) does not matter
    '''
    bestModel = -1
    bestL = float("-inf")

    T, d = mfcc.shape
    M = models[0].omega.shape[0]

    modelLik = []

    for i in range(len(models)):
        myTheta = models[i]
        my_log_Bs = log_Bs(mfcc, myTheta)
        my_log_Lik = logLik(my_log_Bs, myTheta)
        modelLik.append((myTheta, my_log_Lik))
        if my_log_Lik > bestL:
            bestModel = i
            bestL = my_log_Lik

    modelLik.sort(key=lambda x: x[1], reverse=True)
    
    print(models[correctID].name)
    for i in range(k):
        print(modelLik[i][0].name, modelLik[i][1])

    return 1 if (bestModel == correctID) else 0


if __name__ == "__main__":

    trainThetas = []
    testMFCCs = []
    d = 13
    k = 5  # number of top speakers to display, <= 0 if none
    M = 8  # component num
    # # M = 7
    # M = 6
    # # M = 5
    epsilon = 0.0
    # maxIter = 20
    # maxIter = 15
    # maxIter = 10
    maxIter = 5
    i = 0
    # train a model for each speaker, and reserve data for testing
    for subdir, dirs, files in os.walk(dataDir):
        for speaker in dirs:
            # if i > 20:
            #     break
            # i += 1
            # if i > 15:
            #     break
            # i += 1
            # if i > 10:
            #     break
            # i += 1
            print( speaker )

            files = fnmatch.filter(os.listdir( os.path.join( dataDir, speaker ) ), '*npy')
            random.shuffle( files )
            
            testMFCC = np.load( os.path.join( dataDir, speaker, files.pop() ) )
            testMFCCs.append( testMFCC )

            X = np.empty((0,d))
            for file in files:
                myMFCC = np.load( os.path.join( dataDir, speaker, file ) )
                X = np.append( X, myMFCC, axis=0)

            trainThetas.append( train(speaker, X, M, epsilon, maxIter) )

    # evaluate 
    numCorrect = 0;
    for i in range(0,len(testMFCCs)):
        numCorrect += test( testMFCCs[i], i, trainThetas, k ) 
    accuracy = 1.0*numCorrect/len(testMFCCs)
    print(accuracy)

