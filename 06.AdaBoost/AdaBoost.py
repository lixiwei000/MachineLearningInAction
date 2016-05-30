'''
AdaBoost 自适应boosting算法
1.根据原始数据集构建多个数据集,采用随机抽样法
2.boosting通过集中关注被已有的分类器错分的那些数据来重新获得新的分类器
'''

from numpy import *
# 创建简单地数据集
from snowballstemmer.danish_stemmer import lab0


def loadSimpData():
    dataMat = matrix([[1,2.1],[2,1.1],[1.3,1],[1,1],[2,1]])
    classLabel = [1.0,1.0,-1.0,-1.0,1.0]
    return dataMat,classLabel

'''
单层决策树生成函数
通过阀值比较对数据进行分类
'''
def stumpClassify(dataMatrix,dimen,threshVal,threshIneq):
    retArray = ones((shape(dataMatrix)[0],1))       # shape(matrix)[0]计算数组的行数;初始化m*1的矩阵
    if threshIneq == "lt":
        retArray[dataMatrix[:,dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:,dimen] > threshVal] = -1.0
    return retArray
'''
遍历stumpClassify函数所有可能的输入值,找到数据集上最佳的单层决策树
'''
def buildStump(dataArr,classLabels,D):
    dataMatrix = mat(dataArr)           # 创建训练数据集
    labelMat = mat(classLabels).T       # 训练集标签
    m,n = shape(dataMatrix)             # 训练集维度
    numSteps = 10.0                     #
    bestStump = {}                      # 保存给定权重向量D时所得到的最佳单层决策树信息
    bestClasEst = mat(zeros((m,1)))
    minError = inf                      # 最小错误率
    for i in range(n):                  # 遍历所有特征
        rangeMin = dataMatrix[:,i].min()    # 第i个特征的最小值,matrix[i,:]获取第i行所有数据,matrix[:,i]获取第i列数据
        rangeMax = dataMatrix[:,i].max()    # 第i个特征的最大值
        stepSize = (rangeMax - rangeMin) / numSteps # 通过最小值和最大值来确定需要多大的步长
        for j in range(-1 , int(numSteps) + 1):     # 在步长的范围内进行遍历
            for inequal in ['lt','gt']:             # 在大于和小于之间切换不等式
                threshVal = (rangeMin + float(j) * stepSize)    # 设置阀值
                predictedVals = stumpClassify(dataMatrix,i,threshVal,inequal)   # 预测分类结果
                errArr = mat(ones((m,1)))           # 错误向量
                errArr[predictedVals == labelMat] = 0   # 预测结果正确设置为0
                weightedError = D.T * errArr            # 权重,这事Adaboost与分类器交互的地方
                # print("split: dim %d,thresh %.2f,thresh ineqal: %s,the weighted error is %.3f" % (i,threshVal,inequal,weightedError))
                if weightedError < minError:            # 如果当前错误率小,则保存该单层决策树
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump,minError,bestClasEst       # 分别为 最优决策桩(特征,阀值,判定符号),最小错误率,最优化分结果

# dataMat,classLabels = loadSimpData()

# 单层决策树构建测试
# D = mat(ones((5,1)) / 5)
# bestStump,minError,bestClasEst = buildStump(dataMat,classLabels,D)
# print(bestStump,minError)
# print(bestClasEst)

# 基于单层决策树的AdaBoost训练过程
def adaBoostTrainDS(dataArr,classLabels,numIt=40):
    weakClassArr = []
    m = shape(dataArr)[0]                           # 训练集行数
    D = mat(ones((m,1))/m)                          # 初始化数据点权重
    aggClassEst = mat(zeros((m,1)))                 # 初始化 数据点的类别估计累计值
    for i in range(numIt):                          # 循环 numIt次 或 错误率为0 退出
        bestStump,error,classEst = buildStump(dataArr,classLabels,D)# 创建 单层决策树
        print ("D:",D.T)
        alpha = float(0.5*log((1.0-error)/max(error,1e-16)))# 计算 本次单层决策树输出结果的权重
        bestStump['alpha'] = alpha                          # 存储 单层决策树输出结果权重
        weakClassArr.append(bestStump)                      # 存储 单层决策树的参数
        print ("classEst: ",classEst.T)
        expon = multiply(-1*alpha*mat(classLabels).T,classEst) # 计算 下一次迭代中的新权重向量D
        D = multiply(D,exp(expon))                             # 计算 下一次迭代中的新权重向量D
        D = D/D.sum()                                          # 计算 下一次迭代中的新权重向量D

        aggClassEst += alpha*classEst
        print ("aggClassEst: ",aggClassEst.T)
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T,ones((m,1)))       # 将错误的标记为1
        errorRate = aggErrors.sum()/m                                                   # 计算错误率
        print ("total error: ",errorRate,"\n")
        if errorRate == 0.0: break
    return weakClassArr,aggClassEst

# 测试AdaBoost训练结果
# classifierArray = adaBoostTrainDS(dataMat,classLabels,9)
# print(classifierArray)

# AdaBoost分类函数,利用多个训练出来的弱分类器进行分类的函数
def adaClassify(dataToClass,classifierArr):
    dataMatrix = mat(dataToClass)                   # 初始化测试数据
    m = shape(dataMatrix)[0]
    aggClassEst = mat(zeros((m,1)))
    for i in range(len(classifierArr)):             # 遍历弱分类器
        # 分类结果
        classEst = stumpClassify(dataMatrix,classifierArr[i]['dim'],classifierArr[i]['thresh'],classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha']*classEst   # 数据点类别估计累计值,所占权重
        print(aggClassEst)
    return sign(aggClassEst)
# 测试AdaBoost分类函数
# adaClassify([[5,5],[0,0]],classifierArray[0])

# 读取真实数据
def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split("\t"))    # 数据集列数
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split("\t")
        for i in range(numFeat - 1):                        # 遍历每行数据的所有特征
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat
# 真实数据测试
# dataArr,labelArr = loadDataSet("/Users/lixiwei-mac/Documents/IdeaProjects/MachineLearningInAction/06.AdaBoost/data/horseColicTraining2.txt")
# classifierArray = adaBoostTrainDS(dataArr,labelArr,10)
# print(dataArr)
# testArr,testLabelArr = loadDataSet("/Users/lixiwei-mac/Documents/IdeaProjects/MachineLearningInAction/06.AdaBoost/data/horseColicTest2.txt")
# prediction10 = adaClassify(testArr,classifierArray[0])
# errArr = mat(ones((67,1)))
# errRate = errArr[prediction10 != mat(testLabelArr).T].sum()
# print("Error Rate: " , errRate / 67)

'''
非均衡分类问题
我们之前做的分类器,对错误结果都没有进行任何处理,错了就是错了.
掩盖了样例如果被分错的事实.
使用  混淆矩阵  来更好的了解分类中的错误.
'''
# ROC曲线的绘制及AUC计算函数
def plotROC(predStrengths,classLabels):     # 分类器预测强度,
    import matplotlib.pyplot as plt
    cur = (1.0,1.0)
    ySum = 0
    numPosClas = sum(array(classLabels) == 1.0)
    yStep = 1 / float(numPosClas)
    xStep = 1 / float(len(classLabels) - numPosClas)
    sortedIndicies = predStrengths.argsort()
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    for index in sortedIndicies.tolist()[0]:
        if classLabels[index] == 1.0:
            delX = 0
            delY = yStep
        else:
            delX = xStep
            delY = 0
        ax.plot([cur[0],cur[0]-delX],[cur[1],cur[1]-delY],c='b')
        cur = (cur[0]-delX,cur[1]-delY)
    ax.plot([0,1],[0,1],'b--')
    plt.xlabel('假阳率')
    plt.ylabel("真阳率")
    plt.title("病马Adaboost分类器ROC曲线")
    ax.axis([0,1,0,1])
    plt.show()
    print("AUC面积:",ySum*xStep)

dataArr,labelArr = loadDataSet('/Users/lixiwei-mac/Documents/IdeaProjects/MachineLearningInAction/06.AdaBoost/data/horseColicTraining2.txt')
classifierArray,aggClassEst = adaBoostTrainDS(dataArr,labelArr,10)
plotROC(aggClassEst.T,labelArr)
