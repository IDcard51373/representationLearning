# coding=utf-8
import numpy as np
import os
from PIL import Image
import numpy as np
import random
import kits

train_num=30
max_error=1e-5
row_x=30
k_d=60
# 打开数据集
data = np.array(kits.openImages("resources")).T

# 1. 将数据集划分为训练集和测试集；这里前三十个作为训练集 后九个作为测试机
Y_train=data[:,0:train_num-1]
Y_test=data[:,train_num:]


# 2. 在训练集上使用K - SVD算法得到字典，字典大小自行设计或参考课程讲义；
D=kits.k_svd(Y_train,row_x,max_error,k_d)

# 3. 对测试集中的人脸图像进行像素缺失处理；
Y_lossed=kits.loss_pixel(Y_test)


# 4. 在测试集上利用字典得到稀疏表达，然后使用字典重建人脸图像；
representation=kits.omp(D,Y_lossed,row_x)

# 5. 计算重建人脸图像与原始图像的平均误差。
Y_represented=np.dot(D,representation)
r=Y_test-Y_represented
e=[]
for i in range(Y_test.shape[1]):
    e.append(np.linalg.norm(r[i]))

print(e)