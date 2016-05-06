import matplotlib.pyplot as plt
# import DecisionTree as trees

decisionNode = dict(boxstyle="sawtooth", fc="0.8")  # 定义文本框和箭头常量
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")


# 使用文本注解绘制树节点
def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt,  # 绘制带箭头的注解
                            xy=parentPt,
                            xycoords='axes fraction',
                            xytext=centerPt,
                            textcoords='axes fraction',
                            va='center',
                            ha='center',
                            bbox=nodeType,
                            arrowprops=arrow_args)


# 旧的测试绘制节点方法
def createPlot_old():
    fig = plt.figure(1, facecolor='white')  # 创建新图形
    fig.clf()  # 清空绘图区
    createPlot.ax1 = plt.subplot(111, frameon=False)
    plotNode(U'决策节点', (0.5, 0.1), (0.1, 0.5), decisionNode)  # 绘制节点
    plotNode(U'叶节点', (0.8, 0.1), (0.3, 0.8), leafNode)
    plt.show()


# 获取叶子节点的数目
def getNumLeafs(myTree):
    numleafs = 0
    firstStr = list(myTree.keys())[0]  # 第一次划分数据集的类别标签
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            numleafs += getNumLeafs(secondDict[key])
        else:
            numleafs += 1
    return numleafs


# 获取树的层数
def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth:
            maxDepth = thisDepth
    return maxDepth


# 树的信息
def retrieveTree(i):
    listOfTrees = [{'不能浮出水面': {0: 'no', 1: {'有脚蹼': {0: 'no', 1: 'yes'}}}},
                   {'不能浮出水面': {0: 'no', 1: {'有脚蹼': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}
                   ]
    return listOfTrees[i]


# 根据算法结果,绘制完整的决策树
def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString)


def plotTree(myTree, parentPt, nodeTxt):
    numLeafs = getNumLeafs(myTree)  # 树的宽度
    depth = getTreeDepth(myTree)  # 树的深度
    firstStr = list(myTree.keys())[0]  # 第一个
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs)) / 2.0 / plotTree.totalW, plotTree.yOff)
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD  # 减小Y的偏移量,向下画图
    for key in secondDict.keys():
        if type(secondDict[
                    key]).__name__ == 'dict':  # test to see if the nodes are dictonaires, if not they are leaf nodes
            plotTree(secondDict[key], cntrPt, str(key))  # recursion
        else:  # it's a leaf node print the leaf node
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD


# 新的绘制节点函数,创建绘图区
def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plotTree.totalW = float(getNumLeafs(inTree))  # 存储树的宽度
    plotTree.totalD = float(getTreeDepth(inTree))  # 存储树的深度
    plotTree.xOff = -0.5 / plotTree.totalW  # 追踪已绘制节点的位置
    plotTree.yOff = 1.0  # 追踪已绘制节点的位置
    plotTree(inTree, (0.5, 1.0), '')
    plt.show()



# createPlot()

# myTree = retrieveTree(1)
# numleafs = getNumLeafs(myTree)
# print(myTree)
# print(numleafs)

# myTree = retrieveTree(1)
# createPlot(myTree)
# myTree['不能浮出水面'][3] = 'maybe'
# print(myTree)
# createPlot(myTree)

# myDat, labels = trees.createDataSet()
# myTree = retrieveTree(0)
# print(myTree)
# print(labels)
# print(trees.classify(myTree, labels, [0, 1]))

# trees.storeTree(myTree,'classifierStorage.txt')
# localTree = trees.grabTree('classifierStorage.txt')
# print(localTree)

