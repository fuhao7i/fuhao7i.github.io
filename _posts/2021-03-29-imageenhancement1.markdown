---
layout:     post
title:      "Image Enhancement🐽1——AODNet: All-in-One Dehazing Network"
subtitle:   " \"图像去雾\""
date:       2021-03-29 09:13:00
author:     "fuhao7i"
header-img: "img/in-post/imageenhancement.jpg"
catalog: true
tags:
    - Image Enhancement🐽
latex: ture
---

### paper: AOD-Net: All-in-One Dehazing Network

## 1. Physical Model：The atmospheric scattering model

$$
\large I(x) = J(x)t(x)+A(1-t(x))   \tag {1}
$$

其中，$I(x)$是得到的雾图，$J(x)$是场景光辉(清晰的图片)，$A$是全局的光照强度，$t(x)$是传播矩阵，如下所示：

$$
\large t(x) = e^{- \beta d(x)}  \tag 2
$$

其中，%\beta%是大气散射系数，$d(x)$是物体到相机的距离。

根据这个模型，我们进行一个简单的推导，就能得到如何由一个模糊图像得到清晰的图像，从而起到图像增强的效果。

$$
\large J(x) = {\frac{1}{t(x)}}I(x) - A{\frac{1}{t(x)}} + A  \tag 3
$$

$I(x)$已经有了，就是我们的模糊图像，接下来我们只需要依靠神经网络求得$t(x)$和$A$就好了。以前的方法都是单独的估计$t(x)$和$A$的值，但这样并不能使在$J(x)$上重构建的误差最小，以致于模型也不是最优的。这里作者重新构造函数为:

$$
\large J(x) = K(x)I(x) - K(x) + b, where \\
\large K(x) = {\frac{\frac{1}{t(x)}(I(x)-A)+(A-b)}{I(x)-1}} \tag 4
$$

这样$\frac{1}{t(x)}$和$A$就被整合到一个新的变量$K(x)$中了，$b$是一个默认值为1 到常数.

## 2. Model

<img src="https://img-blog.csdnimg.cn/20210329111445369.png" center>

如图所示，模型用了5个输出维度全为3的卷积层，并做了3次规律的堆叠。

`python实现`
```python
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class AODnet(nn.Module):   
    def __init__(self):
        super(AODnet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=6, out_channels=3, kernel_size=5, padding=2)
        self.conv4 = nn.Conv2d(in_channels=6, out_channels=3, kernel_size=7, padding=3)
        self.conv5 = nn.Conv2d(in_channels=12, out_channels=3, kernel_size=3, padding=1)
        self.b = 1

    def forward(self, x):  
        
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x1))
        cat1 = torch.cat((x1, x2), 1)
        x3 = F.relu(self.conv3(cat1))
        cat2 = torch.cat((x2, x3),1)
        x4 = F.relu(self.conv4(cat2))
        cat3 = torch.cat((x1, x2, x3, x4),1)
        k = F.relu(self.conv5(cat3))

        if k.size() != x.size():
            raise Exception("k, haze image are different size!")

        output = k * x - k + self.b
        return F.relu(output)

model = AODnet()

out = model(input)

__call__()
```

## 3. loss

```python
#===== Loss function & optimizer =====
criterion = torch.nn.MSELoss()

if args.cuda:
    criterion = criterion.cuda()

optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=0.0001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=53760, gamma=0.5)
```

## 4. 数据集

输入的是`模糊图像`，标签为`groundtruth清晰图像`。

