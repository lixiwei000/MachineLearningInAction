'''
Logistic回归
概念:
    用一条直线对一些数据点进行拟合,拟合过程称为回归
    Logistic回归的主要思想是:根据数据对分类边界线建立回归公式,以此进行分类.
    "回归"源于"最佳拟合",表示要找到最佳拟合参数集
需要解决的问题:
    找到最佳回归系数,拟合出一个分类函数,用分类函数对输入数据进行分类.

梯度上升法/梯度下降法:
    要找到某函数的最大值,就沿着该函数的梯度方向探寻.(求偏导数)

线性回归分类步骤:
1.求得回归系数theata,
2.求得theata * data 的值,其中θTx=0 即θ0+θ1*x1+θ2*x2=0 称为决策边界即boundarydecision。
3.将2的结果带入阶跃函数,以0.5为分界线进行二分类
'''
from math import exp
from numpy import *
from numpy.random.mtrand import weibull
import matplotlib.pyplot as plt

# Logistic回归梯度上升优化算法
def loadDataSet():
    dataMat = []; labelMat = []
    fr = open('/Users/lixiwei-mac/Documents/IdeaProjects/MachineLearningInAction/04.LogisticRegression/data/testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])               # 每条数据添加一个值为1的特征x0
        labelMat.append(int(lineArr[2]))                                        # 保存每条数据的标签
    return dataMat,labelMat
# 预测函数,阶跃函数 0~1之间
def sigmoid(inX):
    return 1.0/(1+exp(-inX))
# 梯度上升算法
def grandAscent(dataMatIn,classLabels):
    dataMatrix = mat(dataMatIn)                                                 # 将数据转换为numpy矩阵n*3
    labelMat = mat(classLabels).transpose()                                     # 标签矩阵转置   n*1
    m,n = shape(dataMatrix)                                                     # 求矩阵维度
    alpha = 0.001                                                               # 步长
    maxCycles = 500                                                             # 最大循环次数
    weights = ones((n,1))                                                       # 初始化回归系数 n*1
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)                                       #
        error = (labelMat - h)                                                  # 计算真是类别与预测类别的差值
        weights += alpha * dataMatrix.transpose() * error                       # 按照差值的方向调整回归系数
    return weights
# 随机梯度上升算法
def stocGradAscent0(dataMatrix,classLabels):
    m,n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i] * weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]                       # 一次仅用一个系数来更新回归系数
    return weights
# 随机梯度上升算法改进版
def stocGradAscent1(dataMatrix,classLabels,numIter=150):
    m,n = shape(dataMatrix)
    weights = ones(n)
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4/(1.0+j+i) + 0.01                                  # alpha每次迭代都会进行调整
            randIndex = int(random.uniform(0,len(dataIndex)))           # 随机选取样本来更新回归系数
            h = sigmoid(sum(dataMatrix[randIndex] * weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights
# 画出决策边界
def plotBestFit(weights):
    dataMat,labelMat = loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i,1])
            ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1])
            ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1,ycord1,s=30,c='red',marker='s')
    ax.scatter(xcord2,ycord2,s=30,c='green')
    x = arange(-3.0,3.0,0.1)
    y = (-weights[0] - weights[1] * x ) / weights[2]
    ax.plot(x,y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

# 案例:使用病症预测病马死亡率
def classifyVector(inX,weights):
    prob = sigmoid(sum(inX * weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0
def colicTest():
    frTrain = open("/Users/lixiwei-mac/Documents/IdeaProjects/MachineLearningInAction/04.LogisticRegression/data/horseColicTraining.txt")
    frTest = open("/Users/lixiwei-mac/Documents/IdeaProjects/MachineLearningInAction/04.LogisticRegression/data/horseColicTest.txt")
    trainingSet = []
    trainingLabels = []
    for line in frTrain.readlines():                                        # 获取训练集的矩阵和标签
        currLine = line.strip().split("\t")
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainWeights = stocGradAscent1(array(trainingSet),trainingLabels,500)   # 得到训练集的回归系数
    errorCount = 0
    numTestVec = 0.0
    for line in frTest.readlines():                                         # 遍历测试集
        numTestVec += 1.0
        currLine = line.strip().split("\t")
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(array(lineArr),trainWeights)) != int(currLine[21]):   # 使用sigmoid计算测试数据预测结果是否正确
            errorCount += 1
    errorRate = (float(errorCount) / numTestVec)
    print("错误率为:",errorRate)
    return errorRate

def multiTest():
    numTests = 10
    errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print("迭代 %d 次后,平均错误率为: %f" %(numTests,errorSum/float(numTests)))

# dataMat,labelMat = loadDataSet()
# print(dataMat)
# print(labelMat)
# weights = grandAscent(dataMat,labelMat)
# weights = stocGradAscent1(array(dataMat),labelMat,numIter=10000)
# print(weights)
# plotBestFit(weights)
multiTest()