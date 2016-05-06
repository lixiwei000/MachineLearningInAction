'''
决策树相关概念:
0.采用ID3算法划分数据集
1.信息增益:划分数据前后信息发生的变化.获得信息增益最高的特征就是最好的选择.
2.香农熵:集合信息的度量方式.信息的期望值.
构建决策树的关键就在于 选取最佳的特征值,按这个特征值对数据集进行划分.
'''
from math import log
import operator
import TreePlotter as treePlotter
# 计算给定数据集的香农熵
def calcShannoEnt(dataSet):
    numEntries = len(dataSet)                               # 数据集个数
    labelCounts = {}
    for featVec in dataSet:                                 # 遍历每条数据,创建标签字典
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries           # 当前标签出现的概率
        shannonEnt -= prob * log(prob,2)
    return shannonEnt                                       # 熵越高,则混合的数据越多(分类越多)
# 创建数据集
def createDataSet():
    dataSet = [
        [1,1,'yes'],
        [1,1,'yes'],
        [1,0,'no'],
        [0,1,'no'],
        [0,1,'no']
    ]
    labels = ['不能浮出水面',"有脚蹼"]
    return dataSet,labels
# 划分数据集
def splitDataSet(dataSet,axis,value):
    retDataSet = []
    for featVec in dataSet:                                 # 遍历数据集,抽取出去除第axis个特征值为value的新数据
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])         # extend()函数扩展list:a->list->(1,2,3,a),append()函数添加list:a->list->(1,2,3,[a])
            retDataSet.append(reducedFeatVec)
    return retDataSet
# 选择最好的数据集划分方式(计算熵)
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1                       # 获取特征数量
    baseEntropy = calcShannoEnt(dataSet)                    # 计算数据集的熵
    bestInfoGain = 0.0                                      # 默认增益
    bestFeature = -1                                        # 默认选取的最好特征编号
    for i in range(numFeatures):                            # 遍历特征,
        featList = [example[i] for example in dataSet]      # 获取所有数据的第i个特征值
        uniqueVals = set(featList)                          # 特征值去重后集合
        newEntropy = 0.0                                    # 子集默认的熵
        for value in uniqueVals:                            # 按第i个特征值为value进行数据集的划分
            subDataSet = splitDataSet(dataSet,i,value)      # 划分数据集
            prob = len(subDataSet) / float(len(dataSet))    # 子集概率
            newEntropy += prob * calcShannoEnt(subDataSet)  # 子集的熵
        infoGain = baseEntropy - newEntropy                 # 总集合的熵减去新集合的熵结果为信息增益
        if (infoGain > bestInfoGain):                       # 保存增益最大的划分,记录为划分最佳特征
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature
# 按概率选举
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]
# 创建决策树
def createTree(dataSet,labels):
    classList = [example[-1] for example in dataSet]        # 保存所有数据的类别标签
    if classList.count(classList[0]) == len(classList):     # 如果了类别完全相同则停止继续划分
        return classList[0]
    if len(dataSet[0]) == 1:                                # 如果特征用光了,则使用概率最高的类别作为叶子的最终类别
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)            # 选择最优的特征按其进行划分数据集
    bestFeatLabel = labels[bestFeat]                        # 获取标签含义
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])                                   # 删除已经用过的特征
    featValues = [example[bestFeat] for example in dataSet] # 获取所有最佳特征的值
    uniqueValues = set(featValues)                          # 获取去重后的最优特征值
    for value in uniqueValues:                              # 递归构建决策树
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet,bestFeat,value),subLabels)
    return myTree

# 使用决策树的分类函数 获取testVce为值的特征标签
def classify(inputTree,featLabels,testVec):
    firstStr = list(inputTree.keys())[0]                    # 获取第一个分类标签说明
    secondDict = inputTree[firstStr]                        # 获取第一个子集
    featIndex = featLabels.index(firstStr)                  # 将标签字符串转换为索引
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key],featLabels,testVec)
            else:
                classLabel = secondDict[key]
    return classLabel

# 使用pickle模块序列化,存储创建号的决策树对象
def storeTree(inputTree,filename):
    import pickle
    fw = open(filename,'wb')
    pickle.dump(inputTree,fw)
    fw.close()

def grabTree(filename):
    import pickle
    fr = open(filename,"rb")
    return pickle.load(fr)


dataSet,labels = createDataSet()
# shannonEnt = calcShannoEnt(dataSet)
# print(shannonEnt)

# retDataSet = splitDataSet(dataSet,0,1)                      # 取出第0个特征为1的数据,并去掉该特征
# print(dataSet)
# print(retDataSet)

# bestFeature = chooseBestFeatureToSplit(dataSet)
# print(bestFeature)

# myTree = createTree(dataSet,labels)
# print(myTree)

# 使用算法
fr = open('/Users/lixiwei-mac/Documents/IdeaProjects/MachineLearningInAction/DecisionTree/lenses.txt')
lenses = [inst.strip().split('\t') for inst in  fr.readlines()]
lensesLabels = ['age','prescript','astigmatic','tearRage']
lensesTree = createTree(lenses,lensesLabels)
print(lensesTree)
treePlotter.createPlot(lensesTree)

