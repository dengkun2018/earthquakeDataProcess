#-*- coding:utf-8 -*-
__author__ = 'dengkun'

'''
kNN一般流程：
1.收集数据
2.准备数据
3.分析数据
4.训练数据
5.测试数据
6.使用算法
'''

from numpy import *
import operator
from os import listdir

def createDataSet():
    group=array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels=['A','A','B','B']
    return group,labels


def classify0(inX,dataSet,labels,k):
    # 读取矩阵的行数，也就是样本数量，用上面的举例就是4行
    dataSetSize = dataSet.shape[0]
    # 变成和dataSet一样的行数，行数=原来*dataSetSize，列数=原来*1，然后每个特征点和样本的点进行相减
    #tile(A,n)，功能是将数组A重复n次，构成一个新的数组
    #前面用tile，把一行inX变成4行一模一样的（tile有重复的功能，dataSetSize是重复4遍，后面的1保证重复完了是4行，而不是一行里有四个一样的）
    #然后再减去dataSet，是为了求两点的距离，先要坐标相减，这个就是坐标相减
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    # 平方
    sqDiffMat = diffMat ** 2
    # axis=0 按列求和，1为按行求和
    sqDistances = sqDiffMat.sum(axis=1)
    # 开根号，距离就出来了
    distances = sqDistances ** 0.5
    # 按照大小逆序排序，argsort是排序，将元素按照由小到大的顺序返回下标index，比如([3,1,2]),它返回的就是([1,2,0])
    sortedDistIndicies =distances.argsort()
    classCount = {}
    # 选择距离最小的K个点
    for i in range(k):
        # 返回距离（key）对应类别（value）
        voteIlabel = labels[sortedDistIndicies[i]]
        #get是取字典里的元素，如果之前这个voteIlabel是有的，那么就返回字典里这个voteIlabel里的值，如果没有就返回0（后面写的），
        #这行代码的意思就是算离目标点距离最近的k个点的类别，这个点是哪个类别哪个类别就加1
        classCount[voteIlabel]=classCount.get(voteIlabel,0)+1
    # key=operator.itemgetter(1)的意思是按照字典里的第一个排序，{A:1,B:2},要按照第1个（AB是第0个），即‘1’‘2’排序。reverse=True是降序排序
    sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    # 返回类别最多的类别
    return sortedClassCount[0][0]

# 将文本转化为numpy的解析程序
def file2matrix(filename):
    #打开文件
    fr = open(filename)
    arrayLines=fr.readlines()
    # len() 方法返回对象（字符、列表、元组等）长度或项目个数
    numberOfLines = len(arrayLines)         #get the number of lines in the file
    #用法：zeros(shape, dtype=float, order='C')  返回：返回来一个给定形状和类型的用0填充的数组；
    returnMat = zeros((numberOfLines,3))        #prepare matrix to return
    classLabelVector = []                       #prepare labels return
    index = 0
    for line in arrayLines:
        line = line.strip()
        listFromLine = line.split('\t')
        # 接着我们选取前3个元素，将它们存储到特征矩阵中
        returnMat[index,:] = listFromLine[0:3]
        # python语言可以使用索引值-1表示列表中的最后一列元素，利用这种负索引，我们可以很方便地将列表的最后一列存储到向量classLabelVector
        labels = {'didntLike': 1, 'smallDoses': 2, 'largeDoses': 3}  # 新增
        classLabelVector.append(labels[listFromLine[-1]])  # 去掉了int
        # 把该样本对应的标签放至标签集，顺序与样本集对应。 python语言中可以使用-1表示列表中的最后一列元素
        index += 1
    return returnMat,classLabelVector

# 将文本转化为numpy的解析程序
def file2matrix2(filename):
    #打开文件
    fr = open(filename)
    arrayLines=fr.readlines()
    # len() 方法返回对象（字符、列表、元组等）长度或项目个数
    numberOfLines = len(arrayLines)         #get the number of lines in the file
    #用法：zeros(shape, dtype=float, order='C')  返回：返回来一个给定形状和类型的用0填充的数组；
    returnMat = zeros((numberOfLines,3))        #prepare matrix to return
    classLabelVector = []                       #prepare labels return
    index = 0
    for line in arrayLines:
        line = line.strip()
        listFromLine = line.split('\t')
        # 接着我们选取前3个元素，将它们存储到特征矩阵中
        returnMat[index,:] = listFromLine[0:3]
        # python语言可以使用索引值-1表示列表中的最后一列元素，利用这种负索引，我们可以很方便地将列表的最后一列存储到向量classLabelVector
        #labels = {'didntLike': 1, 'smallDoses': 2, 'largeDoses': 3}  # 新增
        classLabelVector.append(int(listFromLine[-1]))
        # 把该样本对应的标签放至标签集，顺序与样本集对应。 python语言中可以使用-1表示列表中的最后一列元素
        index += 1
    return returnMat,classLabelVector


#归一化特征值=(oldValue-min)/(max-min)
def autoNorm(dataSet):
    minVals=dataSet.min(0)
    maxVals=dataSet.max(0)
    ranges=maxVals-minVals
    normDataSet=zeros(shape(dataSet))
    m=dataSet.shape[0]
    normDataSet=dataSet-tile(minVals,(m,1))
    normDataSet=normDataSet/tile(ranges,(m,1))
    return normDataSet,ranges,minVals

#分类器针对约会网站的测试代码
def datingClassTest():
    hoRatio=0.10
    datingDataMat,datingLabels=file2matrix('datingtestSet.txt')
    normMat,ranges,minVals=autoNorm(datingDataMat)
    #shape函数是numpy.core.fromnumeric中的函数，它的功能是查看矩阵或者数组的维数
    #计算测试向量的数量
    m=normMat.shape[0]
    # 训练样本从第m * hoRatio行开始
    numTestVecs=int(m*hoRatio)
    errorCount=0.0
    #待预测向量从0开始到m*hoRatio结束
    for i in range(numTestVecs):
        #normMat[i,:] 为取出mormMat的第i+1行，作为待预测的向量
        #normMat[numTestVecs:m,:]，为训练样本，取出从i+1行开始的m行，这里m可以大于矩阵的总行数
        classifierResult=classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print("the classifier came back with:%d,the real answer is:%d"\
              %(classifierResult,datingLabels[i]))
        if (classifierResult!=datingLabels[i]):
            errorCount+=1.0
    print("the total error rate is:%f"%(errorCount/float(numTestVecs)))

#约会网站预测函数k
def classifyPerson():
    # 结果集合
    resultList = ['not at all','in small doses','in large doses']
    # 输入属性1 花费多少时间打游戏
    percentTats = float(input("percentage of time spent playing video games?"))
    # 输入属性2 一年行走多长
    ffMiles = float(input("frequent flier miles earned per year?"))
    # 输入属性3 消耗多少冰淇凌
    iceCream = float(input("liters of ice cream consumed per year?"))
    # 读取文件
    datingDataMat,datingLabels = file2matrix2('datingTestSet2.txt')
    # 归一化
    normMat,ranges,minVals = autoNorm(datingDataMat)
    # 输入数据（待检测）
    inArr = array([ffMiles,percentTats,iceCream])
    # 输出结果
    classifierResult = classify0((inArr-minVals)/ranges,normMat,datingLabels,3)
    # 打印结果
    print("your will probably like this person: ",resultList[classifierResult -1])

#将32*32的二进制图像矩阵转换为1*1024的向量
def img2vector(filename):
    returnVect=zeros((1,1024))
    fr=open(filename)
    for i in range(32):
        lineStr=fr.readline()
        for j in range(32):
            returnVect[0,32*i+j]=int(lineStr[j])
    return returnVect

#手写数字识别系统的测试代码
def handwritingClassTest():
    #定义一个list，用于记录分类
    hwLabels = []
    #前面的Python os.listdir 可以列出 dir 里面的所有文件和目录，但不包括子目录中的内容。
    #os.walk 可以遍历下面的所有目录，包括子目录。
    trainingFileList = listdir('trainingDigits')
    #求出文件个数
    m = len(trainingFileList)
    #生成m*1024的array，每个文件分配1024个0
    trainingMat = zeros((m,1024))
    #循环，对每一个file
    for i in range(m):
        #当前文件
        fileNameStr = trainingFileList[i]
        #理解这段代码要知道文件的命名方式,这里是这样命名的9_45.txt，9表示分类，45表示第45个。
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        #调用img2vector，将原文件写入trainingMat
        trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)
    #找到testDigits中的文件
    testFileList = listdir('testDigits')
    #计算误差
    errorCount = 0.0
    #多少个文件
    mTest = len(testFileList)
    #遍历test文件
    for i in range(mTest):
        #test文件
        fileNameStr = testFileList[i]
        #分类
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        #转换成1*1024
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        # 调用knn分类
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        #输出
        print ("the classifier came back with:%d, the real anwer is : %d" % (classifierResult, classNumStr))
        #计算误差
        if (classifierResult != classNumStr):
            errorCount += 1.0
    print ("\n the total numbe of error is: %d" % errorCount)
    print ("\nthe total error rate is: %f" % (errorCount/float(mTest)))











