import numpy as np
from scipy import signal

############################################
# activation function and its differential #
############################################
def sigmoid(x):
    return 1./(1. + np.exp(-x))

def diff_sigmoid(x):
    return sigmoid(x)*(1. - sigmoid(x))

######################################
# loss function and its differential #
######################################
def mean_squared_error(y,t):
    return 0.5*np.sum((y-t)**2)

def diff_mean_squared_error(y,t):
    return y-t

########
# main #
########
def cnn_2class():
    rng = np.random

    ########################
    # parameter definition #
    ########################
    numTrainingData = 6
    numTestData = 2
    row = 7
    col = 7
    numFilter = 2
    filterRow = 3
    filterCol = 3
    poolingRow = 2
    poolingCol = 2
    strideRow = poolingRow
    strideCol = poolingCol
    numHiddenNode = 4
    numOutputNode = 1
    learningRate = 1
    epoch = 10000

    ##############
    # load datas #
    ##############
    x = np.zeros((numTrainingData,row,col), float)
    for i in range(numTrainingData):
        fileName = 'data_%d.csv' % i
        x[i] = np.loadtxt(fileName,delimiter=',')
    T = np.zeros((numTestData,row,col), float)
    for i in range(numTestData):
        fileName = 'test_%d.csv' % i
        T[i] = np.loadtxt(fileName,delimiter=',')
    ans = np.loadtxt('training_ans.csv',delimiter=',')



    ##################################
    ##################################
    # convolution and pooling layers #
    ##################################
    ##################################
    
    ##########################################################
    # filters generated randomly and initialize bias to zero #
    ##########################################################

    flt = np.zeros((numFilter,filterRow,filterCol), float)
    for i in range(numFilter):
        flt[i] = rng.randn(filterRow,filterCol)
    flt_b = np.zeros((numFilter,row,col), float)

    #########################
    # convolution procedure #
    #########################
    xf = np.zeros((numTrainingData,numFilter,row,col), float)
    for i in range(numTrainingData):
        for j in range(numFilter):
            xf[i,j] = signal.correlate(x[i],flt[j],'same') + flt_b[j]

    ###################################
    # pooling procedure (max pooling) #
    ###################################
    poolingOutRow = (row-1)/strideRow + 1
    poolingOutCol = (col-1)/strideCol + 1
    xp = np.zeros((numTrainingData,numFilter,poolingOutRow,poolingOutCol), float)
    for i in range(numTrainingData):
        for j in range(numFilter):
            for k in range(poolingOutRow):
                for l in range(poolingOutCol):
                    xp[i,j,k,l] = np.max(xf[i,j,strideRow*k:strideRow*k+poolingRow,strideCol*l:strideCol*l+poolingCol])

    ###########################
    # convert the shape of xp #
    ###########################
    convertedXp = np.zeros((numTrainingData, numFilter*poolingOutRow*poolingOutCol), float)
    for i in range(numTrainingData):
        convertedXp[i] = xp[i].reshape((numFilter*poolingOutRow*poolingOutCol,))



    ##########################
    ##########################
    # fully-connected layers #
    ##########################
    ##########################
    
    ###################################################
    # initialize the weight vector Wh and Wo randomly #
    ###################################################
    Wh = rng.randn(numFilter*poolingOutRow*poolingOutCol, numHiddenNode)
    Wo = rng.randn(numHiddenNode,)

    ############################
    # initialize the bias term #
    ############################
    bh = np.zeros((numHiddenNode), float)
    bo = np.zeros((numOutputNode), float)

    ######################
    # training procedure #
    ######################
    count = 0
    while (count < epoch):
        count += 1
        err = 0
        for i in range(numTrainingData):
            h = sigmoid(np.dot(convertedXp[i],Wh)+bh)
            o = sigmoid(np.dot(h,Wo)+bo)
            L_over_f = diff_mean_squared_error(o,ans[i])
            f_over_o = diff_sigmoid(np.dot(h,Wo)+bo)
            L_over_o = np.dot(f_over_o, L_over_f)
            o_over_Wo = h.T
            subWo = np.dot(o_over_Wo, L_over_o)
            Wo -= learningRate*subWo 
            bo -= learningRate*L_over_o 

            f_over_h = diff_sigmoid(np.dot(convertedXp[i],Wh)+bh)
            h_over_Wh = x[i]
            o_over_h = np.dot(f_over_h, Wo)
            L_over_h = np.dot(o_over_h, L_over_o)
            subWh = np.outer(convertedXp[i], L_over_h)
            Wh -= learningRate*subWh
            bh -= learningRate*L_over_h

            ##################
            # re-calculation #
            ##################
            h = sigmoid(np.dot(convertedXp[i],Wh)+bh)
            o = sigmoid(np.dot(h,Wo)+bo)
            err += mean_squared_error(o,ans[i])

            if (i==numTrainingData-1):
                print(err)

    ##################
    # test procedure #
    ##################
    Tf = np.zeros((numTestData,numFilter,row,col), float)
    for i in range(numTestData):
        for j in range(numFilter):
            Tf[i,j] = signal.correlate(T[i],flt[j],'same') + flt_b[j]

    Tp = np.zeros((numTestData,numFilter,poolingOutRow,poolingOutCol), float)
    for i in range(numTestData):
        for j in range(numFilter):
            for k in range(poolingOutRow):
                for l in range(poolingOutCol):
                    Tp[i,j,k,l] = np.max(Tf[i,j,strideRow*k:strideRow*k+poolingRow,strideCol*l:strideCol*l+poolingCol])

    convertedT = np.zeros((numTestData,numFilter*poolingOutRow*poolingOutCol), float)
    for i in range(numTestData):
        convertedT[i] = Tp[i].reshape((numFilter*poolingOutRow*poolingOutCol,))

    for i in range(numTestData):
        h = sigmoid(np.dot(convertedT[i],Wh)+bh)
        o = sigmoid(np.dot(h,Wo)+bo)
        print(o)

if __name__ == '__main__':
    cnn_2class()    
