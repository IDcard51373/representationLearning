# 批量打开图片
import numpy as np
import os
from PIL import Image
import numpy as np
import random


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
输入 字典D，代表答的y,迭代次数t
输出 表达x
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
        tmp[index] = x.reshape((t, 1))
        tmp = np.array(tmp).reshape(n_d)
        x[:, i] = tmp
    return x

