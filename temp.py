from sklearn.datasets import make_blobs
#scikit中的make_blobs方法常被用来生成聚类算法的测试数据，直观地说，make_blobs会根据用户指定的特征数量、中心点数量、范围等来生成几类数据，这些数据可用于测试聚类算法的效果。
from sklearn.model_selection import train_test_split
#划分测试集合训练集的样本
import numpy as np
import matplotlib.pyplot as plt
#画图函数
X, y = make_blobs(n_samples=500, centers=2,n_features=2)
# 将500个例子划分为两个聚类，它们分别具有两个特征值
y[np.where(y==0)]=-1


xtra, xtes, ytra, ytes = train_test_split(X, y)
xtra = np.c_[xtra, np.ones(xtra.shape[0])]
xtes = np.c_[xtes, np.ones(xtes.shape[0])]
#np.c：按行连接矩阵；
class Perception:
    def __init__(self):
        self.W = None
#初始化权重；
    def sign(self, z):    # sign()
        # return 1 if z>=0 else -1
        return np.where(z>=0, 1, -1)
        # return np.sign(z)   sign在z==0时，输出值为0

    def fit(self, xtra, ytra, samp, learning_rate):
        n_samples, n_feature = xtra.shape
        self.W = np.random.randn(n_feature)
        #给定维度生成0到1之间的数，维度与xtra一致；
        losses=[]
        for i in range(samp):
            ypred = xtra.dot(self.W)
            loss = sum(np.maximum(0, -ytra * ypred)) / n_samples
            #损失函数的计算
            losses.append(loss)
            #将计算得到的损失函数的值添加到losses列表的末尾
            if i % 1 == 0:
                print(f"After {i} epoch, loss is {loss}.")

            fx = np.argmax(np.maximum(0, -ytra*ypred))
            if ytra[fx]*ypred[fx] > 0:
                break
            dw = xtra[fx]*ytra[fx]    # 找出分类错误的样本，更新权重
            self.W += learning_rate*dw

        return self.W, losses


    def predition(self, xtes, ytes):
        y_ = self.sign(xtes.dot(self.W))
        score = len(np.where(y_ == ytes)[0])/len(ytes)*100
        print(f"The test score is {score}%")
        return y_
    #将预测值合真实值进行比较；可以得到模型的泛化能力；
def plot_hy(X, y, Weight):
    xmin = np.min(X[:, 0])
    xmax = np.max(X[:, 0])
    x_hy = np.array([xmin, xmax])
    y_hy = -(x_hy*Weight[0]+Weight[-1])/Weight[1]
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:,0], X[:,1], c=y)
    plt.plot(x_hy, y_hy, c='b', linewidth=2)
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.show()
    
if  __name__== "__main__": 
    P = Perception()
    samp = 500
    lr = 0.1
    Weight, losses = P.fit(xtra, ytra, samp, lr)
    ypred = P.predition(xtes, ytes)
    plot_hy(X, y, Weight)
    plt.plot(range(len(losses)), losses)
    plt.show()
"""
Spyder Editor

This is a temporary script file.
"""

