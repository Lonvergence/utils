import math
import os
import shutil

import ants
import itk
import numpy as np
import SimpleITK as sitk
import skimage


def img_norm(img):
    mx = img.max()
    mn = img.min()

    img = (img - mn) / (mx - mn) * 255
    return img


def adjust_contrast(image, level, window):
    """
    调整图像的对比度和亮度。

    level: 对比度中心
    window: 对比度宽度

    """
    # 计算最低和最高灰度值
    min_val = level - window / 2
    max_val = level + window / 2

    normalized_img = (image - min_val) / (max_val - min_val)

    normalized_img = np.clip(normalized_img, 0, 1)

    adjusted_img = (normalized_img * 255).astype(np.uint8)

    return adjusted_img