from numpy import *
import operator
import matplotlib
import matplotlib.pyplot as plt
from sympy.physics.units import percent
from os import listdir

def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1],[0,0.2]])
    labels = ['A','A','B','B','B']
    return group,labels
# KNN分类算法
def classify0(inX,dataSet,labels,k):
    dataSetSize = dataSet.shape[0]                   # 获取训练数据集维度
    # 计算距离   欧氏距离公式
    diffMat = tile(inX,(dataSetSize,1)) - dataSet    # x1-x0,y1-y0 tile函数:将inX复制成dataSetSize大小的矩阵
    sqDiffMat = diffMat ** 2                         # (x1-x0)^2 , (y1-y0)^2
    sqDistances = sqDiffMat.sum(axis=1)              # (x1-x0)^2 + (y1-y0)^2
    distances = sqDistances ** 0.5                   # 根号 (x1-x0)^2 , (y1-y0)^2
    sortedDistIndicies = distances.argsort()         # 排序->得到升序 "坐标号"
    classCount = {}
    # 选择距离最小的K个点
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]   # 获取排序后第i个元素的标签
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1  # 统计标签出现频率
    # 排序
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]                    # 返回出现频率最高的标签作为预测结果

# 文本转换为NumPy程序
def file2matrix(filename):
    fr = open(filename)
    arrayOfLines = fr.readlines()
    numberOfLines = len(arrayOfLines)               # 获取文件行数
    returnMat = zeros((numberOfLines,3))            # 创建 row*3 维度的矩阵
    classLabelVector = []                           # 每条数据的标签
    index = 0
    for line in arrayOfLines:                       # 遍历每条数据
        line = line.strip()
        listFromLine = line.split('\t')             # 按Tab切割
        returnMat[index,:] = listFromLine[0:3]      # 将前三个值存入到矩阵的第i行
        classLabelVector.append(int(listFromLine[-1]))      # 对应存储标签
        index += 1
    return returnMat,classLabelVector

# 数据归一化
def autoNorm(dataSet):
    minVals = dataSet.min(0)                        # 获取数据集每个属性的最小值
    maxVals = dataSet.max(0)                        # 获取数据集每个属性的最大值
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))             # 初始化一个dataSet大小的0矩阵
    m = dataSet.shape[0]                            # 获取dataSet行维度
    normDataSet = dataSet - tile(minVals,(m,1))     # oldValue - minValue   -- 两者都是矩阵
    normDataSet = normDataSet / tile(ranges,(m,1))  # (oldvalue - minValue) / (maxValue - minValue)
    return normDataSet,ranges,minVals

# 创建测试数据
# group,labels = createDataSet()
# print(classify0([4,3],group,labels,5))

# 将文本转换成矩阵
datingDataMat,datingLabels = file2matrix('/Users/lixiwei-mac/Documents/IdeaProjects/MachineLearningInAction/01.KNN/dataSet/datingTestSet2.txt')
#print(datingDataMat,"\n",datingLabels)

# 图形化输出数据集1
# fig = plt.figure()
# ax = fig.add_subplot(111)
# x轴:每周玩游戏时间比率  y轴:每周吃冰棍数量
# ax.scatter(datingDataMat[:,1],datingDataMat[:,2],15.0*array(datingLabels),15.0*array(datingLabels))
# plt.show() # 并没有清晰的显示分类结果,数据掺杂在一起

# 图形化输出数据集2
# x轴:每年获取的飞行里程数   y轴:每周玩游戏时间比率
# ax.scatter(datingDataMat[:,0],datingDataMat[:,1],15.0*array(datingLabels),15.0*array(datingLabels))
# plt.show()

# 归一化数值 (两点的飞行里程差对结果影响太大,其他两项简直可以忽略,这不是我们想要的)
# normDataSet,ranges,minVals = autoNorm(datingDataMat)
# print(normDataSet)

# tile函数测试
# print(tile([[1,2],[3,4]],4))

# 分类器针对约会网站的测试代码
def datingClassTest():
    hoRatio = 0.10                                      # 训练集抽取比率,作为测试集
    datingDataMat,datingLabels = file2matrix('/Users/lixiwei-mac/Documents/IdeaProjects/MachineLearningInAction/01.KNN/dataSet/datingTestSet2.txt')
    normMat,ranges,minVals = autoNorm(datingDataMat)    # 特征归一化
    m = normMat.shape[0]                                # 行维度
    numTestVecs = int(m * hoRatio)                      # 测试集数量
    errorCount = 0.0
    for i in range(numTestVecs):                        # 抽取第i个测试集
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3) # 预测
        print("The classifier came back with: %d, the real answer is %d" % (classifierResult,datingLabels[i]))
        if (classifierResult != datingLabels[i]):
            errorCount += 1.0
    print("错误率为 : %f" % (errorCount/float(numTestVecs)))

datingClassTest()

# 约会网站预测函数
def classifyPerson():
    resultList = ['一点也不喜欢','一般般吧','非常喜欢']
    percentTats = float(input("玩游戏花费时间比率:"))
    ffMiles = float(input("每年旅游里程数:"))
    iceCream = float(input("每年吃冰激凌公升数:"))
    datingDataMat,datingLabels = file2matrix('/Users/lixiwei-mac/Documents/IdeaProjects/MachineLearningInAction/01.KNN/dataSet/datingTestSet2.txt')
    normMat,ranges,minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles,percentTats,iceCream])
    classifierResult = classify0((inArr - minVals)/ranges,normMat,datingLabels,3) # 需要将测试数据也归一化
    print("你对这个人的印象是:" , resultList[classifierResult - 1])
# classifyPerson()

# 手写识别系统
def img2vector(filename):
    returnVect = zeros((1,1024))                        # 创建1*1024的0矩阵
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

# print(img2vector("/Users/lixiwei-mac/Documents/IdeaProjects/MachineLearningInAction/dataSet/digits/testDigits/0_13.txt"))

# 测试算法:识别手写数字
def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('/Users/lixiwei-mac/Documents/IdeaProjects/MachineLearningInAction/KNN/dataSet/digits/trainingDigits')
    m = len(trainingFileList)                           # 训练集文件数
    trainingMat = zeros((m,1024))                       # 创建1*1024训练集0矩阵
    for i in range(m):                                  # 将训练集转换为符合格式的向量
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels .append(classNumStr)
        trainingMat[i,:] = img2vector('/Users/lixiwei-mac/Documents/IdeaProjects/MachineLearningInAction/KNN/dataSet/digits/trainingDigits/%s' % fileNameStr)
    testFileList = listdir('/Users/lixiwei-mac/Documents/IdeaProjects/MachineLearningInAction/KNN/dataSet/digits/testDigits')
    mTest = len(testFileList)
    errorCount = {}
    for k in range(3,10):
        error = 0.0;
        for i in range(mTest):
            fileNameStr = testFileList[i]
            fileStr = fileNameStr.split('.')[0]
            classNumStr = int(fileStr.split('_')[0])
            vectorUnderTest = img2vector('/Users/lixiwei-mac/Documents/IdeaProjects/MachineLearningInAction/KNN/dataSet/digits/testDigits/%s' % fileNameStr)
            classifierResult = classify0(vectorUnderTest,trainingMat,hwLabels,3)        # 预测
            print("分类器预测结果:%d,真实结果:%d"%(classifierResult,classNumStr))
            if (classifierResult != classNumStr):
                error += 1.0
        errorCount[k] = error
    for i in errorCount.keys():
        print("\n参数k=%d错误总数:%d"%(i,errorCount[i]))
        print("\n参数k=%d错误率:%f"%(i,errorCount/float(mTest)))
# handwritingClassTest()
'''
k近邻算法的缺点:
    1.必须保存全部的数据集,如果训练数据集很大,必须使用大量的存储空间
    2.由于必须对数据集中的每个数据计算距离值,实际使用时肯呢过非常耗时
'''