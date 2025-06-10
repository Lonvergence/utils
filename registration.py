import math
import os
import shutil

import ants
import itk
import numpy as np
import SimpleITK as sitk
import skimage
from loguru import logger


@logger.catch()
def reg_pipeline(fixed, moving, type_of_transform, *to_transforms, 
                 registration_kwargs={}, apply_transform_kwargs={}):
    """
    fixed: 固定图像
    moving: 浮动图像
    type_of_transform: 变换类型
    mask: mask
    to_transforms: 需要施加形变的图像
    """
    breakpoint()
    mytx = ants.registration(fixed=fixed,
                             moving=moving,
                             type_of_transform=type_of_transform,
                             **registration_kwargs)  

    registered_img = mytx["warpedmovout"]

    result = [ants.apply_transforms(fixed=to_transform,
                                    moving=to_transform,
                                    transformlist=mytx["fwdtransforms"],
                                    interpolator="nearestNeighbor",
                                    **apply_transform_kwargs) 
              for to_transform in to_transforms]

    return registered_img, result

@logger.catch()
def reg_pipeline_reverse(fixed, moving, type_of_transform, *to_transforms):
    """
    反向配准
    """
    mytx = ants.registration(fixed=moving,
                             moving=fixed,
                             type_of_transform=type_of_transform)

    whichtoinvert = None
    if type_of_transform == "Affine":
        whichtoinvert = [True]
    result = [ants.apply_transforms(fixed=to_transform,
                                    moving=to_transform,
                                    transformlist=mytx["invtransforms"],
                                    interpolator="nearestNeighbor",
                                    whichtoinvert=whichtoinvert) for to_transform in to_transforms]

    return mytx["warpedfixout"], result
