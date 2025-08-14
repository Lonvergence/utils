import math
import os
import shutil

import ants
import itk
import numpy as np
import SimpleITK as sitk
import skimage
from loguru import logger

from collections import Counter, defaultdict
from copy import deepcopy

def check_shape_equal(*mats):
    mat0 = mats[0]
    if not all([m.shape == mat0.shape for m in mats[1:]]):
        raise ValueError('Input images must have the same dimensions.')
    return

def sitk_read(path):
    img = sitk.GetArrayFromImage(sitk.ReadImage(path))
    return img


def sitk_write(img, path):
    sitk.WriteImage(sitk.GetImageFromArray(img), path)


# 计时器
def timer(func):
    from time import time

    def warp(*args, **kwargs):
        ts = time()
        result = func(*args, **kwargs)
        te = time()
        print(f"{func.__name__} cost time: {te - ts} s")

        return result

    return warp



def clear_dir(path: str):
    if not os.path.exists(path):
        return

    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"删除{file_path}时出错：{e}")


def get_file_paths(dirpath: str):
    res = []
    paths = [os.path.join(dirpath, p) for p in os.listdir(dirpath)]
    for p in paths:
        if os.path.isfile(p):
            res.append(p)
        elif os.path.isdir(p):
            res += get_file_paths(p)
    return res


class Accumulator:
    def __init__(self, length):
        self.length = length
        self.accumulator = np.zeros((self.length,))
        self.nums = 0

    def __call__(self, *x):
        x = np.array(x)

        assert x.shape == self.accumulator.shape

        self.accumulator += x
        self.nums += 1

    def get_result(self):
        if self.nums == 0:
            return self.accumulator
        return self.accumulator / self.nums

    def clear(self):
        self.__init__(length=self.length)

    def change_length(self, length):
        self.__init__(length)



class RegionMap:
    def __init__(self, json_path):
        import json
        with open(json_path, 'r') as f:
            self.cfg = json.load(f)
        self._all_attr = self._get_all_attr()
    
    def _get_all_attr(self):
        return self.cfg.keys()
    
    def get_cfg_by_attr(self, attr, value):
        if attr not in self._all_attr:
            raise ValueError(f"Attribute {attr} not found in the configuration.")
        cfg = self.cfg
        def recur(cfg):
            if cfg.get(attr) == value:
                return cfg
            for child in cfg.get("children", []):
                result = recur(child)
                if result:
                    return result
            return None
        return recur(cfg)

    
    def get_all_sub(self, attr1, value, attr2):
        # 得到attr1为value的脑区下面所有子脑区的attr2
        res = []
        cfg = self.get_cfg_by_attr(attr1, value)

        def recur(cfg, attr2, res):
            res.append(cfg[attr2])
            for child in cfg.get("children", []):
                recur(child, attr2, res)
        recur(cfg, attr2, res)
        return res

    def cal_region_areas(self, arr):
        # 递归计算所有区域的大小

        cfg = self.cfg
        dis = Counter(arr.flatten())
        def recur(cfg, dis):
            if cfg["id_ytw"] not in dis.keys():
                dis[cfg["id_ytw"]] = 0
            
            for child in cfg["children"]:
                res = recur(child, dis)
                dis[cfg["id_ytw"]] += res

            return  dis[cfg["id_ytw"]]
        
        recur(cfg, dis)

        return dis




if __name__ == "__main__":
    region_map = RegionMap("./add_id_ytw.json")
    print(region_map.get_cfg_by_attr("name", "ventral tegmental decussation"))
    
