import math
import os
import shutil

import ants
import itk
import numpy as np
import SimpleITK as sitk
import skimage

def norm_img(AVGT):
    AVGT = AVGT.astype(np.float64)
    AVGT = np.rint((AVGT-np.min(AVGT)) / (np.max(AVGT)-np.min(AVGT)) * 255)
    AVGT = AVGT.astype(np.uint8)
    return AVGT

def N4_bias_correction(img):
    raw_img = img
    raw_img = norm_img(raw_img).astype(np.float64)
    raw_img = sitk.GetImageFromArray(raw_img)
    correction_img = sitk.N4BiasFieldCorrection(raw_img, raw_img > 0)  # 需要图片是64位浮点类型
    correction_img = sitk.GetArrayFromImage(raw_img)
    return correction_img


def rescale_intensity(img):
    corr = img
    # RESCALE TO 8 BIT
    scale_limit = np.percentile(corr, (99.999))
    corr = skimage.exposure.rescale_intensity(corr, in_range=(0, scale_limit), out_range='uint8')
    img = np.copy(corr)

    # rescale intensity based on mean/median of tissue
    img_temp = np.copy(img)
    scale_thres = skimage.filters.threshold_otsu(img_temp)
    img_temp[img_temp < scale_thres] = 0
    nz_mean = np.mean(img_temp[img_temp > 0].flatten())
    print(nz_mean)
    scale_fact = 120 / nz_mean
    img = img * scale_fact
    print(np.max(img))
    img[img > 255] = 255
    img = img.astype('uint8')
    return img


def Clahe_3D(img):
    auto = img
    auto_temp_hor = np.copy(auto)
    auto_temp_cor = np.copy(auto)
    auto_temp_sag = np.copy(auto)

    for h in range(auto_temp_cor.shape[1]):
        temp_img = auto_temp_cor[:, h, :]

        temp_img[0:2, 0:2] = 250
        temp_img[3:5, 3:5] = 0

        clahe_im = skimage.exposure.equalize_adapthist(temp_img,
                                                       kernel_size=(int(temp_img.shape[0] / 3), int(temp_img.shape[1] / 6)),
                                                       clip_limit=0.01, nbins=255)
        clahe_im[0:2, 0:2] = 0

        clahe_im = clahe_im * 255
        clahe_im[clahe_im < 0] = 0
        clahe_im = np.uint8(clahe_im)
        auto_temp_cor[:, h, :] = clahe_im

    for h in range(auto_temp_hor.shape[0]):
        temp_img = auto_temp_hor[h, :, :]

        temp_img[0:2, 0:2] = 250
        temp_img[3:5, 3:5] = 0

        clahe_im = skimage.exposure.equalize_adapthist(temp_img,
                                                       kernel_size=(int(temp_img.shape[0] / 3), int(temp_img.shape[1] / 6)),
                                                       clip_limit=0.01, nbins=255)
        clahe_im[0:2, 0:2] = 0

        clahe_im = clahe_im * 255
        clahe_im[clahe_im < 0] = 0
        clahe_im = np.uint8(clahe_im)
        auto_temp_hor[h, :, :] = clahe_im

    for h in range(auto_temp_sag.shape[2]):
        temp_img = auto_temp_sag[:, :, h]

        temp_img[0:2, 0:2] = 250
        temp_img[3:5, 3:5] = 0

        clahe_im = skimage.exposure.equalize_adapthist(temp_img,
                                                       kernel_size=(int(temp_img.shape[0] / 3), int(temp_img.shape[1] / 6)),
                                                       clip_limit=0.01, nbins=255)
        clahe_im[0:2, 0:2] = 0

        clahe_im = clahe_im * 255
        clahe_im[clahe_im < 0] = 0
        clahe_im = np.uint8(clahe_im)
        auto_temp_sag[:, :, h] = clahe_im

    # combine angles
    clahe_all = np.zeros((auto_temp_sag.shape[0], auto_temp_sag.shape[1], auto_temp_sag.shape[2], 3))
    clahe_all[:, :, :, 0] = auto_temp_hor
    clahe_all[:, :, :, 1] = auto_temp_cor
    clahe_all[:, :, :, 2] = auto_temp_sag
    clahe_all_mean = np.mean(clahe_all, 3)

    # combine with original volume
    clahe_final = np.zeros((auto_temp_sag.shape[0], auto_temp_sag.shape[1], auto_temp_sag.shape[2], 2))
    clahe_final[:, :, :, 0] = clahe_all_mean
    clahe_final[:, :, :, 1] = auto

    clahe_final = np.mean(clahe_final, 3)
    clahe_final = np.uint8(clahe_final)
    return clahe_final



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