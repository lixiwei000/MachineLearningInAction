'''
支持向量机
'''

import random
from numpy import *
# SMO算法中的辅助函数
def loadDataSet(fileName):
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split("\t")
        dataMat.append([float(lineArr[0]),float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat,labelMat
# 随机选择一个不等于i的下标
def selectJrand(i,m):
    j = i
    while (j == i):
        j = int(random.uniform(0,m))
    return j
# 调整大于H或小于L的alpha值
def clipAlpha(aj,H,L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj

'''
简化版SMO算法
参数:
    数据集.类别标签.常数C.容错率.退出前最大循环次数

'''
def smoSimple(dataMatIn,classLabels,C,toler,maxIter):
    dataMatrix = mat(dataMatIn)                         # 数据特征列表转换为numpy矩阵 m*3
    labelMat = mat(classLabels).transpose()             # 类别转换为 m*1 矩阵
    b = 0
    m,n = shape(dataMatrix)
    alphas = mat(zeros(m,1))                            # 初始化 m*1 系数矩阵
    iter = 0
    while (iter < maxIter):                             # 对alpha进行迭代更新
        alphaPairsChanged = 0                           # 用来记录alpha是否优化过
        for i in range(m):
            fXi = float(multiply(alphas,labelMat).T * (dataMatrix*dataMatrix[i,:].T)) + b   # 预测的标签结果
            Ei = fXi - float(labelMat[i])               # 预测结果与真实的误差
            if ((labelMat[i] * Ei < -toler) and (alphas[i] < C ) or (labelMat[i] * Ei > toler) and (alphas[i] > 0)): # 如果差值超过容错率,则对alpha进行更新
                j = selectJrand(i,m)
                fXj = float(multiply(alphas,labelMat).T * (dataMatrix * dataMatrix[j,:].T)) + b
                Ej = fXj - float(labelMat[j])
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                if (labelMat[i] != labelMat[j]):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C+alphas[j] - alphas[i])
                else:
                    L = max(0,alphas[j] + alphas[i] - C)
                    H = min(C,alphas[j] + alphas[i])
                if L == H:
                    print("L==H")
                    continue
                eta = 2.0 * dataMatrix[i,:] * dataMatrix[i,:].T - \
                      dataMatrix[j,:] * dataMatrix[j,:].T -\
                      dataMatrix[j,:] * dataMatrix[j,:].T
                if eta >= 0:
                    print("eta >=0")
                    continue
                alphas[j] -= labelMat[j] * (Ei - Ej) / eta
                alphas[j] = clipAlpha(alphas[j],H,L)
                if (abs(alphas[j] - alphaJold < 0.00001)):
                    print("J not moving enough")
                    continue
                alphas[i] += labelMat[j] * labelMat[i] * (alphaJold - alphas[j])
                b1 = b - Ei - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i,:] * dataMa


dataArr , labelArr = loadDataSet("/Users/lixiwei-mac/Documents/IdeaProjects/MachineLearningInAction/05.SVM/dataSet/testSet.txt")
print(labelArr)