---
layout:     post
title:      "Competition🐳1——车道线检测识别与标注"
subtitle:   " \"中国软件杯, 车道线的手动标注\""
date:       2021-01-11 12:47:00
author:     "fuhao7i"
header-img: "img/in-post/competition.jpg"
catalog: true
tags:
    - Competition🐳
---

> 车道线的检测识别就是检测出来交通场景中的各种车道线，并返回它们的准确位置信息。目前常见的解决方法有：  
>> 1. 机器学习模型自动检测识别 
>> 2. 利用OpenCv的传统计算机视觉算法进行自动检测识别  

> 这两种方法都可以完成这项任务，但是由于对车道线位置信息的精度要求，再加上整个程序的实时性性能要求，我最终采用的方案是手动标注。因为我的项目应用场景是交通路口摄像头下的道路情况。所以一般我们只需要
>> 1. 在安装程序的时候标注一次，以后就可以不用再标注了；
>> 2. 手动标注精确度可以达到100%；
>> 3. 利用更少的计算资源。

## 1. 提取背景.py

提取视频或摄像头的第一帧作为我们手动标注的图像。

```python
import cv2
vidcap = cv2.VideoCapture('/Users/apple/Documents/二叶/目标追踪/yolov3_deepsort/data/video/直线马路.mp4')
success,image = vidcap.read()
n=1
while n < 30:
	success, image = vidcap.read()
	n+=1
imag = cv2.imwrite('fff.png',image)
if imag ==True:
	print('ok')
```

## 2. 车道线标定.py

斑马线标定顺序从左上角顺时针标定好4个点。

```python
import cv2
from PIL import Image
from pylab import *
import csv
import codecs

img = cv2.imread('fff.png')

cishu = 1
sx = []
bm = []
im = array(Image.open('fff.png'))
ion()
imshow(im)
# 实线标定
for cs in range(cishu):

    print('Please click 2 points')
    x = ginput(2) # 获取两个鼠标点击坐标
    print('you clicked:',x)
    sx.append(x)

# 斑马线标定（只标定了一处斑马线）
print('Please click 4 points')
x = ginput(4) # 获取四个鼠标点击坐标
print('you clicked:',x)
bm.append(x)

ioff()
show()
print(im.shape)
jinzhi = []
banmaxian = []
def shixian(x1,y1,x2,y2):
    if x1 == x2:
        k = -999
        b = 0
    else:
        k = (y2-y1)/(x2-x1)
        b = y1 - x1 * k
        # k = int(k)
        # b = int(b)
    return k,b
#data1 = [{'x1':int(x[0][0]),'y1':int(x[0][1]),'x2':int(x[1][0]),'y2':int(x[1][1])}]
for i in sx:
    x = {}
    x1 = int(i[0][0])
    y1 = int(i[0][1])
    x2 = int(i[1][0])
    y2 = int(i[1][1])
    k, b = shixian(x1,y1,x2,y2)
    #cv2.rectangle(img, (x1+15,y1), (x2-15,y2), (0,0,255), -1)
    if y1 > y2:
        yy = y2
        xx = x2
        y2 = y1
        x2 = x1
        y1 = yy
        x1 = xx
    if k != 0:
        for xxx in range(y1,y2):
            xq = (xxx-b)/k
            xq = int(xq)
            cv2.rectangle(img, (xq+15,xxx), (xq-15,xxx), (0,0,255), -1)
    else:
        for xxx in range(x1,x2):
            yq = b
            cv2.rectangle(img, (xxx,yq+15), (xxx,yq-15), (0,0,255), -1)
    x['k'] = k
    x['b'] = b
    x['x1'] = x1
    x['x2'] = x2
    x['y1'] = y1
    x['y2'] = y2
    print('k:',k,'b:',b)
    jinzhi.append(x)



print(jinzhi)


# data2 = [{'x1':400, 'y1':0,'x2':800,'y2':0,
#           'x3':400, 'y3':800, 'x4':800, 'y4':800}]
for i in bm:
    x = {}
    x1 = int(i[0][0])
    y1 = int(i[0][1])
    x2 = int(i[1][0])
    y2 = int(i[1][1])
    y3 = int(i[2][1])
    k, b = shixian(x1,y1,x2,y2)
    c = y3 - y1
    x['k'] = k
    x['b'] = b
    x['c'] = c
    x['x1'] = x1
    x['x2'] = x2
    x['y1'] = y1
    x['y2'] = y2

    cv2.rectangle(img, (x1,y1+c), (x2,y2), (0,255,0), 4)
    banmaxian.append(x)

print(banmaxian)

# 将标定好的图像进行保存显示
cv2.imwrite('001_new.png', img)

# 将返回信息写入txt文件，方便后面读取使用
with open("shixian.txt", 'w') as f:
    for s in jinzhi:
        f.write(str(s) + '\n')

with open("banmaxian.txt", 'w') as f:
    for s in banmaxian:
        f.write(str(s) + '\n')
```