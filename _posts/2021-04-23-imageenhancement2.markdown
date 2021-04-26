---
layout:     post
title:      "Image Enhancement🐽2——metrics for picture quality"
subtitle:   " \"SSIM(structural similarity), PSNR(Peak Signal to Noise Ratio)\""
date:       2021-04-23 09:03:00
author:     "fuhao7i"
header-img: "img/in-post/imageenhancement.jpg"
catalog: true
tags:
    - Image Enhancement🐽
latex: ture
---

# 1. SSIM: Structural similarity

# 2. PSNR: Peak Signal to Nosie Ratio

# 3. implement

```python
import skimage
from skimage.metrics import structural_similarity as ssim
import numpy as np 
from PIL import Image

im1 = Image.open('./uw1.png')
im2 = Image.open('./gt1.png')

im1 = np.array(im1)
im2 = np.array(im2)

# diff = im1 - im2
# mse = np.mean(np.square(diff))
# psnr = 10 * np.log10(255 * 255 / mse)

# print(psnr)

pnsr = skimage.metrics.peak_signal_noise_ratio(im1, im2, data_range=255)

ssim = skimage.metrics.structural_similarity(im1, im2, data_range=255, multichannel=True)

print(psnr, ssim)
```

# Reference

1. [图像质量评价指标之 PSNR 和 SSIM](https://zhuanlan.zhihu.com/p/50757421)