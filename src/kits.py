# 批量打开图片
import numpy as np
import os
from PIL import Image
#源目录

def openImages(path):
    data=[]
    os.chdir(path)
    for i in os.listdir(os.getcwd()):
        im=Image.open(i)
        data.append(list(im.getdata()))
    return data
