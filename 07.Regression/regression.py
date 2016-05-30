'''
预测数值型数据:回归
与之前的线性回归类似,目标是找到系数向量w,预测结果就是    Y = XT * w
'''


# 数据导入
from numpy import *
def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split("\t")) - 1
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split("\t")
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat

def standRegres(xArr,yArr):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    xTx = xMat.T * xMat
    if linalg.det(xTx) == 0.0:              # 计算行列式  如果|A| = 0 那么A不不可逆
        print("这个矩阵不存在逆矩阵")
        return
    ws = xTx.I * (xMat.T * yMat)            # 系数的计算公式
    return ws
# 画出数据点分布和最佳拟合曲线
xArr,yArr = loadDataSet("/Users/lixiwei-mac/Documents/IdeaProjects/MachineLearningInAction/07.Regression/data/ex0.txt")
ws = standRegres(xArr,yArr)     # 2 * 1
# 绘制数据样本
# xMat = mat(xArr)                # 200 * 2
# yMat = mat(yArr)                # 1 * 200
# yHat = xMat * ws                # 内积
# import matplotlib.pyplot as plt
# fig = plt.figure()
# ax = fig.add_subplot(211)
# ax2 = fig.add_subplot(212)
# ax.scatter(xMat[:,1].flatten().A[0],yMat.T[:,0].flatten().A[0])
# ax2.scatter(xMat[:,1].flatten().A[0],yMat.T[:,0].flatten().A[0])
# # plt.show()
#
# # 绘制最佳拟合曲线
# xCopy = xMat.copy()
# yHat = xCopy * ws               # 如果不排序,那么绘制出的拟合曲线没法看
# xCopy.sort(0)
# yHat1 = xCopy * ws
# ax.plot(xCopy[:,1],yHat)
# ax2.plot(xCopy[:,1],yHat1)
# plt.show()

# 计算相关系数
# print(corrcoef(yHat.T,yMat))        # 需要转置,以保证两个向量都是行向量
'''
局部加权线性回归
之前的线性回归有欠拟合的现象,使用局部加权线性回归,给待预测点附近的每个点富裕一定的权重
'''
def lwlr(testPoint,xArr,yArrm,k=1.0):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    m = shape(xMat)[0]
    weights = mat(eye(m))               # 创建m阶对角矩阵,为每个数据点初始化了一个权重
    for j in range(m):
        diffMat = testPoint - xMat[j,:]  # 随着样本点与待测点距离的递增,权重将以指数级衰减
        weights[j,j] = exp(diffMat * diffMat.T / (-2.0*k**2)) # 高斯核公式,参数k控制衰减速度,距离越远权重越低,越近权重越高
    xTx = xMat.T * (weights * xMat)
    if linalg.det(xTx) == 0.0:
        print("该矩阵不可逆")
        return
    ws = xTx.I * (xMat.T * (weights * yMat))    # 计算回归系数
    return testPoint * ws

def lwlrTest(testArr,xArr,yArr,k=1.0):
    m = shape(testArr)[0]
    yHat = zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i],xArr,yArr,k)
    return yHat
# yPre1 = lwlr(xArr[0],xArr,yArr,1.0)
# yPre2 = lwlr(xArr[0],xArr,yArr,0.001)
# print(yPre1)
# print(yPre2)
# yHat1 = lwlrTest(xArr,xArr,yArr,1.0)
# yHat2 = lwlrTest(xArr,xArr,yArr,0.01)
# yHat3 = lwlrTest(xArr,xArr,yArr,0.003)
# # print(yHat)
# # 局部加权线性回归 拟合效果
# xMat = mat(xArr)
# srtInd = xMat[:,1].argsort(0)
# xSort = xMat[srtInd][:,0,:]
# import matplotlib.pyplot as plt
# # 创建画布
# fig = plt.figure()
# ax1 = fig.add_subplot(311)
# ax2 = fig.add_subplot(312)
# ax3 = fig.add_subplot(313)
# # 画拟合曲线
# ax1.plot(xSort[:,1],yHat1[srtInd],color="red")
# ax2.plot(xSort[:,1],yHat2[srtInd],color="blue")
# ax3.plot(xSort[:,1],yHat3[srtInd],color="yellow")
# # 画数据点
# ax1.scatter(xMat[:,1].flatten().A[0],mat(yArr).T.flatten().A[0],s=2,c='red')
# ax2.scatter(xMat[:,1].flatten().A[0],mat(yArr).T.flatten().A[0],s=2,c='red')
# ax3.scatter(xMat[:,1].flatten().A[0],mat(yArr).T.flatten().A[0],s=2,c='red')
# plt.show()

'''
示例:预测鲍鱼的年龄
'''
