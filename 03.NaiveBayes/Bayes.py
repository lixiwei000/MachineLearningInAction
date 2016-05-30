'''
朴素贝叶斯
0.概念
    贝叶斯决策理论:
        如果p1(x,y)>p2(x,y)那么类别为1
1.条件概率
    在B的前提下发生A
    p(A|B) = p(AB) / p(B)
    已知p(A|B) 求p(B|A)
        p(B|A) = p(A|B)p(B)/p(A)
2.使用条件概率进行分类,贝叶斯分类器
    p1(x,y) 表示(x,y)属于类别1的概率
    但贝叶斯决策理论需要计算p(c1|x,y),即:给定x,y表示的数据点,那么该数据点来自类别c1的概率是多少?
    应用到贝叶斯准侧得到:
        p(c|x,y) = p(x,y|c)p(c)/p(x,y)
        如果p(c1|x,y) > p(c2|x,y) 那么属于类别1
3.使用Bayes进行文档分类
    概述:把文档中的每个词的出现or不出现作为特征,因此具有大量的特征.
        如果特征之间相互独立,即特征或单词出现的可能性与它和其他单词的相邻没有关系,
        假设有1000个样本,每个样本3个特征,那么样本就可以从 N^1000 降到 1000N
    Naive:"朴素"贝叶斯中的朴素,就是假设各个特征在不同位置出现的概率是相同的,其实这种假设不正确,但是效果还不错
        举例: 宫保鸡丁好吃/好玩   显然,特征"好吃"和"好玩"在宫保鸡丁后面出现的概率并不相同,但是我们仍然假设相同.
        朴素也假设每个特征同等重要.这个假设也有问题.难道1000字的留言我要1000个特征才能判断么,其实几十个字就够了.
优点:
    与之前学习过的KNN和DecisionTree相比:
        Bayes仅需要很少量的数据集就能进行分类,不需要KNN进行大量的计算.
        如果特征值很少,DecisionTree需要使用最优特征对数据进行数据集划分,但结果一定不是很理想(因为特征太少)
    然而Bayes采用概率的方法来进行分类预测,不需要大量的数据和很多特征.???

使用Python进行文本分类
社区评论的敏感词过滤

朴素贝叶斯分类器使用:
1.获取单词源,制作词汇表
2.每一组词源为作为训练集,抽取不分词源作为测试集
3.使用训练集,创建 不同类别 的单词 分值矩阵 ,词频越高,其本分类的代表性越强
4.使用测试集,对3得到的不同类别的分值矩阵进行向量加,之后整合为多个类别概率分值
5.比较4得到分值,分值高德作为预测结果
'''
from email import feedparser

from nltk import classify
from numpy import *
import operator
# 词表到向量的转换函数


def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]                                # 1 爱表侮辱性文字, 0 代表正常言论
    return postingList,classVec                             # prostingList:分词后的文档集合 classVec:类别标签的集合
# 创建词汇表,每个词汇是唯一的
def createVocabList(dataSet):
    vocabSet = set([])                                      # 创建空集
    for document in dataSet:
        vocabSet = vocabSet | set(document)                 # 创建两个集合的并集
    # return list(vocabSet).sort()
    return sorted(vocabSet,key=operator.itemgetter(0),reverse=False)
# vocabList:去重后词汇表   inputSet:需要进行检测的列表   返回词典大小的标识矩阵,1表示有,0表示没有
def setOfWords2Vec(vocabList,inputSet):
    returnVec = [0] * len(vocabList)                        # 创建一个其中所含元素都为0的向量
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1            # 标识单词是否在集合中出现的位置
        else:
            print("单词: %s 不在我的词汇表中" % word)
    return returnVec                                        # 最终将一组单词转换为一组数字
# 朴素贝叶斯词袋模型
def bagOfWords2VecMN(vocabList,inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        returnVec[vocabList.index(word)] += 1
    return returnVec
'''
朴素贝叶斯分类器训练函数
trainMatrix:文档矩阵
trainCatrgory:每篇文档类别标签构成的向量
'''
def trainNB0(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix)                         # 文档总数
    numWords = len(trainMatrix[0])                          # 词典长度
    pAbusive = sum(trainCategory) / float(numTrainDocs)     # 计算侮辱性文档的概率(class=1) ,trainCategory是人工标记的
    # p0Num = zeros(numWords)                                 # 初始化概率,为了计算p(wi|c1)和p(wi|c0),初始化分子分母
    # p1Num = zeros(numWords)
    p0Num = ones(numWords)                                  # 改:防止一个概率为0的数乘以任何数都为0
    p1Num = ones(numWords)

    # p0Denom = 0.0
    # p1Denom = 0.0
    p0Denom = 2.0
    p1Denom = 2.0
    for i in range(numTrainDocs):                           # 遍历文档
        if trainCategory[i] == 1:                           # 如果当前文档属于侮辱性文档
            p1Num += trainMatrix[i]                         # 向量相加,词袋模型显示单词词频
            p1Denom += sum(trainMatrix[i])                  # 敏感词文档中单词总数
        else:                                               # 如果当前文档属于正常言论
            p0Num += trainMatrix[i]                         # 向量想家,词袋模型显示单词词频
            p0Denom += sum(trainMatrix[i])                  # 正常言论中单词总数
    # p1Vect = p1Num / p1Denom                                # 对每个元素做除法
    # p0Vect = p0Num / p0Denom
    p1Vect = log(p1Num/p1Denom)                             # 改:防止下溢出
    p0Vect = log(p0Num/p0Denom)
    return p0Vect,p1Vect,pAbusive                           # 输出单词在 正常/侮辱 文档中出现的概率,概率越大越有代表性

# 朴素贝叶斯分类函数
def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0

def testingNB():
    listOfPosts,listClasses = loadDataSet()
    myVocabList = createVocabList(listOfPosts)
    trainMat = []
    for postinDoc in listOfPosts:
        trainMat.append(setOfWords2Vec(myVocabList,postinDoc))
    p0V,p1V,pAb = trainNB0(array(trainMat),array(listClasses))
    testEntry = ['love','my','dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList,testEntry))
    print("testEntry 被分类为",classifyNB(thisDoc,p0V,p1V,pAb))
    testEntry = ['stupid','garbage']
    thisDoc = array(setOfWords2Vec(myVocabList,testEntry))
    print("testEntry 被分类为",classifyNB(thisDoc,p0V,p1V,pAb))


#4.5测试
# testingNB()

# listOfPosts,listCLasses = loadDataSet()
# myVocabList = createVocabList(listOfPosts)
# print(myVocabList)
# myVec = setOfWords2Vec(myVocabList,listOfPosts[0])
# print(myVec)

# Bayes分类器测试
# listOfPosts,listCLasses = loadDataSet()
# myVocabList = createVocabList(listOfPosts)
# trainMat = []
# for postinDoc in listOfPosts:
#     trainMat.append(setOfWords2Vec(myVocabList,postinDoc))          # 将单词转换成数字,标记出现的单词在词典中的位置
# p0V,p1V,pAb = trainNB0(trainMat,listCLasses)
# print(myVocabList)
# print(p0V)
# print("\n")
# print(p1V)
# print("\n")
# print(pAb)
'''
['I', 'ate', 'buying', 'cute', 'dalmation', 'dog', 'food', 'flea', 'garbage', 'has', 'how', 'him', 'help', 'is', 'licks', 'love', 'my', 'maybe', 'mr', 'not', 'please', 'posting', 'problems', 'park', 'quit', 'steak', 'stupid', 'so', 'stop', 'take', 'to', 'worthless']
[ 0.04166667  0.04166667  0.          0.04166667  0.04166667  0.04166667
  0.          0.04166667  0.          0.04166667  0.04166667  0.08333333
  0.04166667  0.04166667  0.04166667  0.04166667  0.125       0.
  0.04166667  0.          0.04166667  0.          0.04166667  0.          0.
  0.04166667  0.          0.04166667  0.04166667  0.          0.04166667
  0.        ]


[ 0.          0.          0.05263158  0.          0.          0.10526316
  0.05263158  0.          0.05263158  0.          0.          0.05263158
  0.          0.          0.          0.          0.          0.05263158
  0.          0.05263158  0.          0.05263158  0.          0.05263158
  0.05263158  0.          0.15789474  0.          0.05263158  0.05263158
  0.05263158  0.10526316]


0.5
'''

'''
示例:使用朴素贝叶斯过滤垃圾邮件
1.收集数据:提供文本文件
2.准备数据:将文本文件解析成词条向量
3.分析数据:检查词条正确性
4.训练算法:使用trainNB0计算概率矩阵
5.测试算法:使用classifyNB
6.使用算法:对一组文档进行分类
'''
# 文本解析,将字符串解析为单词列表
def textParse(bigString):
    import re
    listOfTokens = re.split(r'\W*',bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 3]            # 返回词列表

def spamTest():
    docList = []; classList = []; fullText = []
    basePath = "/Users/lixiwei-mac/Documents/IdeaProjects/MachineLearningInAction/03.NaiveBayes/email/"
    for i in range(1,26):                                                   # 导入并解析文本
        wordList = textParse(open(basePath + 'spam/%d.txt' % i,encoding="gbk").read())     # 转换为词列表
        docList.append(wordList)                                            # 加入文档组
        fullText.extend(wordList)                                           # 加入单词组
        classList.append(1)                                                 # 加入类别组
        wordList = textParse(open(basePath + "ham/%d.txt" % i,encoding="gbk").read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)                                    # 获取词汇表(去重)
    trainingSet = list(range(50))
    testSet = []
    for i in range(10):                                                     # 随机构建训练集
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])                                         # 被选中作为测试的数据,需要从训练集中删除
    trainMat = []; trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(setOfWords2Vec(vocabList,docList[docIndex]))        # 构造训练集单词出现的01矩阵
        trainClasses.append(classList[docIndex])                            # 构造训练集分类矩阵
    p0V,p1V,pSpam, = trainNB0(array(trainMat),array(trainClasses))
    errorCount = 0
    for docIndex in testSet:                                                # 对测试集进行分类
        wordVector = setOfWords2Vec(vocabList,docList[docIndex])
        if classifyNB(array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
            errorCount += 1
            print("错误词表:",docList[docIndex])
    print("错误率为:" , float(errorCount)/len(testSet))

spamTest()

'''
示例:使用朴素贝叶斯分类器从个人广告中获取区域倾向
'''
# ny = feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
# print(len((ny['entries'])))
# 计算出现频率最高的30个单词
def calcMostFreq(vocabList,fullText):
    import operator
    freqDict = {}
    for token in vocabList:
        freqDict[token] = fullText.count(token)
    sortedFreq = sorted(freqDict.items(),key=operator.itemgetter(1),reverse=True)
    return sortedFreq[:30]

import feedparser
def loadWords(feed1,feed0):
    docList = []; classList = []; fullText = []
    minLen = min(len(feed1['entries']),len(feed0['entries']))
    for i in range(minLen):                                                 # 遍历两种数据源,保存单词
        wordList = textParse(feed1['entries'][i]['summary'])                # 获取单词列表
        docList.append(wordList)                                            # 添加一组单词
        fullText.extend(wordList)                                           # 添加出现的所有单词
        classList.append(1)                                                 # 添加类别
        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)                                    # 创建词汇表,去重
    top30Words = calcMostFreq(vocabList,fullText)                           # 获取频率最高的30个单词,去掉他们,提高正确率
    # for pairW in top30Words:
    #     if pairW[0] in vocabList:
    #         vocabList.remove(pairW[0])
    trainingSet = list(range(2*minLen))                                     # 创建训练集
    testSet = []                                                            # 创建测试集
    for i in range(20):                                                     # 随机抽取20个数据加入测试集
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat = []; trainClasses = []
    for docIndex in trainingSet:                                            # 遍历训练集,
        trainMat.append(bagOfWords2VecMN(vocabList,docList[docIndex]))      # 创建基于词典的词袋稀疏矩阵
        trainClasses.append(classList[docIndex])                            # 添加对应的类别
    p0V,p1V,pSpam = trainNB0(array(trainMat),array(trainClasses))           # 训练模型,计算出概率矩阵
    errorCount = 0
    for docIndex in testSet:
        wordVector = bagOfWords2VecMN(vocabList,docList[docIndex])
        if classifyNB(array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
            errorCount += 1
            print("错误分类:",docList[docIndex])
    print("错误率:" ,float(errorCount) / len(testSet))
    return vocabList,p0V,p1V

# ny = feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
# sf = feedparser.parse('http://sfbay.craigslist.org/stp/index.rss')
# vocabList,pSF,pNY = loadWords(ny,sf)

# 最具表征性德词汇显示函数
def getTopWords(ny,sf):
    import operator
    vocabList,p0V,p1V = loadWords(ny,sf)
    topNY = []; topSF = []
    for i in range(len(p0V)):
        if p0V[i] > -6.0:
            topSF.append((vocabList[i],p0V[i]))
        if p1V[i] > -6.0:
            topNY.append((vocabList[i],p1V[i]))
    sortedSF = sorted(topSF,key=lambda pair:pair[1],reverse=True)
    print("SF*SF*SF*SF*SF*SF*SF*SF*SF*SF*SF*SF*SF*SF*SF*SF*SF*SF*SF*SF*SF*SF*SF*")
    for item in sortedSF:
        print(item[0])
    sortedNY = sorted(topNY,key=lambda pair : pair[1],reverse=True)
    print("NY*NY*NY*NY*NY*NY*NY*NY*NY*NY*NY*NY*NY*NY*NY*NY*NY*NY*NY*NY*NY*NY*NY*")
    for item in sortedNY:
        print(item[0])
# getTopWords(ny,sf)
