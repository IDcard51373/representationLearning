# 批量打开图片
import numpy as np
import os
from PIL import Image
import numpy as np
import random

max_iter_times=40
"""
打开目录下的图片
输入：路径
输出：数据集data
"""
def openImages(path):
    data=[]
    os.chdir(path)
    for i in os.listdir(os.getcwd()):
        im=Image.open(i)
        data.append(list(im.getdata()))
    return data



"""
omp算法
输入 字典D(shape=m*k)待表达的y(shape=m*n),迭代次数t
输出 表达x(shape=k*n)
"""
def omp(D,y,t):

    n_d=D.shape[1]  # n_d:D的列数，即条目数量
    n_y=y.shape[1]  # n_y：y的个数
    l_dy=D.shape[0]  # l_dy:D的行数，理论上D和y的行数相同
    x=np.zeros((n_d,n_y))
    for i in range(n_y):
        yi = y[:, i]
        r = yi  # r:残差
        index = []  # 初始化
        for k in range(t):  # 开始迭代
            ar = np.fabs(np.dot(D.T, r))
            lambda_k = np.argmax(ar)  # 找到最大投影对应的下标
            index.append(lambda_k)  # 加入选择的基下标
            # 支撑集a中加入选择的基
            if k == 0:
                a = D[:, lambda_k].reshape(l_dy, 1)
            else:
                a = np.concatenate((a, D[:, lambda_k].reshape(l_dy, 1)), axis=1)
            x_k = np.dot(np.linalg.pinv(a), y)  # 重新计算表达
            r = y - np.dot(a, x_k)  # 得到新的残差
        temp = np.zeros((n_d, 1))
        temp[index] = x_k
        x[:, i] = np.array(temp).reshape(n_d)  # 格式要转换一下才能赋值进去
    return x



"""
k_svd算法
输入 Y(一组原始数据，shape=m*n)，S(最后表达的x的零范数)，E(误差)，K(字典条目数量，x的长度)
输出 字典D（shape=m*K）
"""
def k_svd(Y,S,E,K):
    m=Y.shape[0]
    n=Y.shape[1]
    X=np.zeros((K,n))

    #字典初始化
    D= np.random.random((m,K))
    #字典、原始数据归一化
    for i in range(K):
        norm = np.linalg.norm(D[:, i])
        mean = np.sum(D[:, i]) / m
        D[:, i] = (D[:, i] - mean) / norm
    for i in range(n):
        norm = np.linalg.norm(Y[:, i])
        mean = np.sum(Y[:, i]) / m
        Y[:, i] = (Y[:, i] - mean) / norm
    # 迭代
    for j in range(max_iter_times):
        x =omp(D, Y, S)
        e = np.linalg.norm(Y - np.dot(D, X))
        if e < E:
            break
        # 逐行调整
        for i in range(K):





#######################################################



        for j in range(iter_times):
            # X = linear_model.orthogonal_mp(D, Y, None)
            X = OMP(D, Y, T)
            # X = cs_omp(Y,D)
            e = np.linalg.norm(Y - np.dot(D, X))
            f.write(str('%s' % e) + '\n')
            print(str('第%s次迭代，误差为：%s' % (j, e)) + '\n')
            if e < err:
                break
            for k in range(K):
                index = np.nonzero(X[k, :])[0]
                if len(index) == 0:
                    continue
                D[:, k] = 0
                R = (Y - np.dot(D, X))[:, index]
                u, s, v = np.linalg.svd(R, full_matrices=False)
                D[:, k] = u[:, 0].T
                X[k, index] = s[0] * v[0, :]
        return D

