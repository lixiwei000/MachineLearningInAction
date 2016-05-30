from numpy import *
def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split("\t")
        fitLine = [float(x) for x in curLine]
        dataMat.append(fitLine)
    return dataMat

def distEclud(vecA,vecB):
    return sqrt(sum(power(vecA - vecB,2)))

def randCent(dataSet,k):
    n = shape(dataSet)[1]
    centroids = mat(zeros((k,n)))
    for j in range(n):
        minJ = min(dataSet[:,j])
        rangeJ = float(max(dataSet[:,j]) - minJ)
        centroids[:,j] = minJ + rangeJ * random.rand(k,1)
    return centroids

dataMat = mat(loadDataSet("/Users/lixiwei-mac/Documents/IdeaProjects/MachineLearningInAction/09.K-Means/data/testSet2.txt"))
# print(min(dataMat[:,0]))
# print(min(dataMat[:,1]))
# print(max(dataMat[:,0]))
# print(max(dataMat[:,1]))
# print("随机质心\n",randCent(dataMat,2))
# print("距离:",distEclud(dataMat[0,:],dataMat[1,:]))

# K均值聚类
def kMeans(dataSet,k,distMeas=distEclud,createCent=randCent):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m,2)))      # 初始化 数据分类结果
    centroids = createCent(dataSet,k)       # 创建随机质心
    clusterChanged = True                   # 迭代结束标识
    while clusterChanged:                   # 使用新的质心位置重新进行分类,知道所有指点位置不再改变为止
        clusterChanged =False
        for i in range(m):              # 对所有数据进行分类
            minDist = inf
            minIndex = -1
            for j in range(k):              # 寻找最近的质心
                distJI = distMeas(centroids[j,:],dataSet[i,:])  # 计算第i个数据与第j个质心的据林
                if distJI < minDist:        # 保存最小距离
                    minDist = distJI        # 数据距离被归类质心的距离
                    minIndex = j            # 数据分类的质心编号
            if clusterAssment[i,0] != minIndex: # 如果数据i之前的分类和现在的分类不相同,则设置数据改变标识为true,需要再次迭代
                clusterChanged = True
            clusterAssment[i,:] = minIndex,minDist**2   # 给第i个数据点分类,保存结果
        # print (centroids)
        for cent in range(k):                           # 重新计算质心位置
            ptsInClust = dataSet[nonzero(clusterAssment[:,0].A == cent)[0]] # 获取所有点属于cent类别的角标后抽取角标下的数据
            centroids[cent,:] = mean(ptsInClust,axis=0) # 对cent类别的数据,沿X轴方向进行均值计算
    return centroids,clusterAssment
# 展示数据点
def drawPoints(dataMat,dataLabel,k):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    colors = ['red','blue','green','black','yellow','pink','orange']
    for cent in range(k):
        ptsInClust = dataMat[nonzero(dataLabel == cent)[0]]
        ax.scatter(ptsInClust[:,0],ptsInClust[:,1],random.randint(5,100),color=colors[cent])
    plt.show()

# centroids,clustAssing = kMeans(dataMat,4)
# print("\n质心坐标\n",centroids)
# print("\n分来结果:\n",clustAssing[:,0])
# drawPoints(dataMat,clustAssing[:,0],4)

# 二分 K-Means 算法
def biKmeans(dataSet,k,distMeas=distEclud):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m,2)))                  # 初始化分类结果
    centroid0 = mean(dataSet,axis=0).tolist()[0]        # 将整个数据集作为一个族,计算质心位置
    centList = [centroid0]
    for j in range(m):                                  # 初始化保存所有质心的数组
        clusterAssment[j,1] = distMeas(mat(centroid0),dataSet[j,:]) ** 2    # 保存数据点距离质心的距离
    while (len(centList) < k):
        lowestSSE = inf
        for i in range(len(centList)):
            ptsInCurrCluster = dataSet[nonzero(clusterAssment[:,0].A == i)[0],:]    # 选出当前类别的数据点
            centroidMat,splitClusterAss = kMeans(ptsInCurrCluster,2,distMeas)       # 对当前数据集进行k=2聚类
            sseSplit = sum(splitClusterAss[:,1])    # 计算平方误差
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:,0].A != i)[0],1]) # 计算不属于当前类别数据点的平方误差
            print("sseSplit,notSplit:",sseSplit,sseNotSplit)
            if (sseSplit + sseNotSplit < lowestSSE):    # 如果误差之和小于最小误差,则保存本次划分
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClusterAss.copy()
                lowestSSE = sseSplit + sseNotSplit
        bestClustAss[nonzero(bestClustAss[:,0].A == 1)[0],0] = len(centList)    # 更新族的分配结果
        bestClustAss[nonzero(bestClustAss[:,0].A == 0)[0],0] = bestCentToSplit
        print("The bestCentToSplot is :",bestCentToSplit)
        print("The len of bestClustAss is :",len(bestClustAss))
        centList[bestCentToSplit] = bestNewCents[0,:]
        centList.append(bestNewCents[1,:])
        clusterAssment[nonzero(clusterAssment[:,0].A == bestCentToSplit)[0],:] = bestClustAss
    return centList,clusterAssment

# 测试二分K-Means
# centList,myNewAssments = biKmeans(dataMat,3)
# print(centList)
# print()
# drawPoints(dataMat,myNewAssments[:,0],3)

# 示例: 对地图上的店进行聚类
import urllib.request as req
import urllib.parse as pa
import json
def geoGrab(stAddress,city):
    apiStem = "http://where.yahooapis.com/geocode?"
    params = {}
    params['flags'] = 'J'
    params['appid'] = 'YD-9G7bey8_JXxQP6rxl.fBFGgCdNjoDMACQA--'
    params['location'] = '%s %s' % (stAddress,city)
    yahooApi = apiStem + pa.urlencode(params)
    print(yahooApi)
    c = req.urlopen(yahooApi)
    return json.loads(c.read())
geoGrab('1 VA Center','Augusta,ME')