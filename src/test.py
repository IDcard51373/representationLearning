# coding=utf-8
import numpy as np
import os
from PIL import Image
import numpy as np
import random
import kits
import copy

# 一些训练参数
# 固定参数
train_num = 30
max_error = 1e-5
k_d = 300
img_h = 192
img_w = 168

# 可调参数
loss_p = 50  # 像素缺失率
sparse_x = 10  # 表达的稀疏度

# 实验记录
psnr = []


# 打开数据集
path = "resources"
data = []
os.chdir(path)
for i in os.listdir(os.getcwd()):
    im = Image.open(i)
    data.append(np.array(im.getdata()).reshape(img_h, img_w))

# 1. 将数据集划分为训练集和测试集；这里前三十个作为训练集 后九个作为测试机
Y_train = data[:train_num]
Y_test = data[train_num:]

# 2. 在训练集上使用K - SVD算法得到字典，字典大小自行设计或参考课程讲义；
# 训练集切块
train_patchs = np.array(kits.img_split(Y_train[0]))
for i in range(1, len(Y_train)):
    patchs = kits.img_split(Y_train[i])
    train_patchs = np.concatenate((train_patchs, patchs), axis=1)
train_patchs = train_patchs[:, np.random.randint(0, high=train_patchs.shape[1] - 1, size=img_h * img_w // 64)]
# 字典学习
D = kits.k_svd(train_patchs, sparse_x, max_error, k_d)

# 3. 对测试集中的人脸图像进行像素缺失处理；
Y_lossed = kits.loss_pixel(Y_test, loss_p)

# 4. 在测试集上利用字典得到稀疏表达，然后使用字典重建人脸图像；
Y_rec = []
for i in range(len(Y_test)):
    kits.to_img(Y_lossed[i]).convert('RGB').save("..\\out\\" + str(i) + "_loss.jpg")
    kits.to_img(Y_test[i]).convert('RGB').save("..\\out\\" + str(i) + "_ori.jpg")
    lossed_patchs = kits.img_split(Y_lossed[i])
    rec_patchs = copy.deepcopy(lossed_patchs)
    for j in range(rec_patchs.shape[1]):
        temp_patch = rec_patchs[:, j]
        index = np.nonzero(temp_patch)[0]
        if index.shape[0] == 0:
            continue
        l2norm = np.linalg.norm(temp_patch[index])
        mean = np.sum(temp_patch) / index.shape[0]
        normed_patch = (temp_patch - mean) / l2norm
        representation = kits.omp(D[index, :], normed_patch[index].T, sparse_x)
        rec_patchs[:, j] = np.fabs(((D.dot(representation) * l2norm) + mean).reshape(rec_patchs.shape[0]))
    rec_img = kits.img_integrate(rec_patchs, Y_lossed[i].shape)
    rec_img = rec_img.astype(np.uint8)
    Y_rec.append(rec_img)
    kits.to_img(rec_img).convert('RGB').save("..\\out\\" + str(i) + "_rec.jpg")
# 5. 计算重建人脸图像与原始图像的平均误差。并输出图像
for i in range(len(Y_test)):
    p=kits.psnr(Y_test[i], Y_rec[i])
    psnr.append(p)
    print(str(i) + "次实验的psnr为：" + str(p))
kits.record(psnr,'psnr')
"""
note:
要切片！否则归一化之后就是0了
测试集只选一部分去学习字典，否则计算量太大，耗时太长
对性能误差的测试采用psnr
浅拷贝深拷贝
"""
