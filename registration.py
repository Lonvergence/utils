import math
import os
import shutil

import ants
import itk
import numpy as np
import SimpleITK as sitk
import skimage
from loguru import logger
from utils import sitk_read, sitk_write

import torch
import torch.nn as nn
import torch.nn.functional as nnf

import scipy.io as sio
import nibabel as nib

@logger.catch()
def reg_pipeline(fixed, moving, type_of_transform, *to_transforms, 
                 registration_kwargs={}):
    """
    fixed: 固定图像
    moving: 浮动图像
    type_of_transform: 变换类型
    mask: mask
    to_transforms: 需要施加形变的图像
    """
    mytx = ants.registration(fixed=fixed,
                             moving=moving,
                             type_of_transform=type_of_transform,
                             random_seed=42,
                             **registration_kwargs)  

    registered_img = mytx["warpedmovout"]

    result = [ants.apply_transforms(fixed=to_transform,
                                    moving=to_transform,
                                    transformlist=mytx["fwdtransforms"],
                                    interpolator="nearestNeighbor") 
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

def build_map(
    ave,
    type_of_transform,
    label_num, # 0 表示没有label, >1 为label的数量
    save_dir,
    *data_list,
    outprefix="ave",
    threshold=0.45
):
    """
    做平均脑
    """

    if isinstance(ave, str):
        ave = sitk_read(ave)

    new_ave = np.zeros(ave.shape, dtype=np.float32)
    ave_multi_region = None if label_num == 0 else {str(i): np.zeros(ave.shape, 'float') for i in range(1, label_num + 1)}
    ave_label = np.zeros(ave.shape, dtype=np.uint8)

    # reg
    ave = ants.from_numpy(ave)
    for brain in data_list:

        brain = sitk_read(brain)
        brain = ants.from_numpy(brain)
        label = None
        if label_num:
            label = sitk_read(brain.replace(".nii.gz", "_label.nii.gz"))
            label = ants.from_numpy(label)
            warepd_brain, [warped_label, ] = reg_pipeline(ants.from_numpy(ave), ants.from_numpy(brain), type_of_transform, label)
            warped_brain = warped_brain.numpy()
            warped_label = warped_label.numpy()
        else:
            warped_brain, _ = reg_pipeline(fixed=ants.from_numpy(ave), moving=ants.from_numpy(brain), type_of_transform=type_of_transform)
            warped_brain = wapred_brain.numpy()
        
        new_ave += warped_brain

        if label_num:
            for i in range(1, label_num + 1):
                ave_multi_region[str(i)] += (warped_label == i) * i
            
    # ave
    data_len = len(data_list)
    new_ave /= data_len
    for i in range(1, label_num + 1):
        value = ave_multi_region[str(i)]
        value /= datasets_size

        value[value < threshold * i] = 0
        value[value != 0] = np.max(value)

        # 形态学后处理
        import skimage
        value = skimage.morphology.closing(value)
        value = skimage.morphology.opening(value)
        ave_label[value != 0] = i
    
    # save
    os.makedirs(save_dir)
    sitk_write(new_ave, os.path.join(save_dir, f"{outprefix}.nii.gz"))
    if label_num:
        sitk_write(ave_label, os.path.join(save_dir, f"{outprefix}_label.nii.gz"))

def ants_mat_to_4x4(ants_mat_fn):
    """
    Read in and convert the .mat file from ANTs format to a 4x4
    matrix.
    """
    # read in .mat file
    _dict = sio.loadmat(ants_mat_fn)
    # define transformation from LPS to RAS
    lps2ras = np.diag([1, 1, 1])
    # get rotation, translation, and center
    rot = _dict['AffineTransform_float_3_3'][0:9].reshape((3,3))
    trans = _dict['AffineTransform_float_3_3'][9:12]
    center = _dict['fixed']

    r_offset = (- np.dot(rot, center) + center + trans).T * [1, 1, 1]
    r_rot = np.dot(np.dot(lps2ras, rot), lps2ras)
    # r_offset = -np.dot(rot, center) + center + trans
    # r_rot = rot
    
    data = np.eye(4)
    data[0:3, 3] = r_offset
    data[:3, :3] = r_rot
    
    return data

        
def cal_field_intensity(file_path, shape=None):
    """
    计算形变场强度
    """
    name = os.path.basename(file_path).split('.')[-1]    
    if name == 'mat':
        # 线性形变
        assert shape is not None, 'shape is None!'
        
        mat44 = ants_mat_to_4x4(file_path)
        
        coordinates= np.indices(shape).transpose(1,2,3,0)
        homogeneous_coordinates = np.concatenate((coordinates, np.ones((*shape, 1))), axis=-1)
        transformed_coordinates = mat44 @ homogeneous_coordinates.reshape(-1, 4).T
        transformed_coordinates = transformed_coordinates.T.reshape(*shape, 4)
        # 相对位移
        transformed_coordinates = transformed_coordinates[..., :3] - coordinates
        field_intensity = np.linalg.norm(transformed_coordinates, axis=-1)
        
        return field_intensity
    else:
        # 非线性形变
        field = nib.load(file_path)
        field = field.get_fdata()
        field = np.squeeze(field)
        field_intensity = np.linalg.norm(field, axis=-1)
        
        return field_intensity        
    
    
    

class ANTS_Tool:
    def __init__(self):
        pass

    @staticmethod
    def reg_pipeline(fixed, moving, 
                     type_of_transform,
                     *to_transforms,
                     **registration_kwargs):
        """
        save_name_list: fix, mov, to_transform1, to_transform2, ...
        """
        mytx = ants.registration(fixed=fixed,
                                moving=moving,
                                type_of_transform=type_of_transform,
                                **registration_kwargs)  

        registered_img = mytx["warpedmovout"]

        result = [ants.apply_transforms(fixed=to_transform,
                                        moving=to_transform,
                                        transformlist=mytx["fwdtransforms"],
                                        interpolator="nearestNeighbor") 
                for to_transform in to_transforms]

        return registered_img, result, mytx

    @staticmethod
    def save_ants_params(
        mytx, 
        save_dir, 
        save_fwd=True,
        save_inv=False,
        ):
        '''
        保存配准参数
        '''

        if not save_fwd and not save_inv:
            raise ValueError("At least one of save_fwd or save_inv must be True.")

        cfg = {}
        ANTS_Tool._mkdirs(os.path.join(save_dir, "transforms"))

        if save_fwd:
            cfg["fwdtransforms"] = list()
            for x in mytx["fwdtransforms"]:
                new_path = os.path.join(save_dir, "transforms", os.path.basename(x))
                ANTS_Tool._cp(x, new_path)
                cfg["fwdtransforms"].append(new_path)
        
        if save_inv:
            cfg["invtransforms"] = list()
            cfg["whichtoinvert"] = list()
            for x in mytx["invtransforms"]:
                new_path = os.path.join(save_dir, "transforms", os.path.basename(x))
                ANTS_Tool._cp(x, new_path)
                cfg["invtransforms"].append(new_path)
                if os.path.basename(x).endwith(".mat"):
                    cfg["whichtoinvert"].append(True)
                else:
                    cfg["whichtoinvert"].append(False)
        
        cfg_path = os.path.join(save_dir, "ants_params.json")
        with open(cfg_path, 'w') as f:
            f.write(json.dumps(cfg, indent=4))
        
        return cfg_path
    
    @staticmethod
    def reg_with_params(data_dir, data_list, interpolator_list="nearestNeighbor", use_fwd=True):
        '''
        使用已有配准参数
        '''
        with open(os.path.join(data_dir, "ants_params.json"), 'r') as f:
            cfg = json.load(f)

        if use_fwd:
            transforms = cfg.get("fwdtransforms", [])
            whichtoinvert = None
        else:
            transforms = cfg.get("invtransforms", [])
            whichtoinvert = cfg.get("whichtoinvert", None)
        if not transforms:
            raise ValueError("No transforms found in the configuration.")

        return ANTS_Tool._apply_transforms(transforms, 
                                     data_list,
                                     interpolator_list, 
                                     whichtoinvert,)

    @staticmethod
    def _mkdirs(d):
        os.makedirs(d, exist_ok=True)
        
    @staticmethod
    def _cp(src, dst):
        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)

    @staticmethod
    def _sitk_write(x, path):
        sitk.WriteImage(sitk.GetImageFromArray(x), path)

    @staticmethod
    def _apply_transforms(transformlist, 
                          to_transforms,
                          interpolator_list="nearestNeighbor", 
                          whichtoinvert=None,
                          ):
        if isinstance(interpolator_list, str):
            interpolator_list = [interpolator_list] * len(to_transforms)
        return [ants.apply_transforms(fixed=to_transform,
                                  moving=to_transform,
                                  transformlist=transformlist,
                                  interpolator=interpolator_list[i],
                                  whichtoinvert=whichtoinvert) 
            for i, to_transform in enumerate(to_transforms)]

class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, mode='bilinear'):
        super().__init__()

        self.mode = mode

    def forward(self, src, flow):
        # new locations
        grid = self.get_grid(src.shape[2:])
        grid = grid.to(src.device)
        new_locs = grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return nnf.grid_sample(src, new_locs, align_corners=True, mode=self.mode)

    def get_grid(self, size):
        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)
        return grid
    

def ANTs_transformation_to_dvf(ants_file : str, shape=None):
    # return [D, H, W, 3]
    name = os.path.basename(ants_file).split('.')[-1]
    if name == 'mat':
        assert shape is not None, 'shape is None!'
        
        mat44 = ants_mat_to_4x4(ants_file)
        # mat44_inv = np.linalg.inv(mat44)
        
        coordinates= np.indices(shape).transpose(1,2,3,0)
        homogeneous_coordinates = np.concatenate((coordinates, np.ones((*shape, 1))), axis=-1)
        transformed_coordinates = mat44 @ homogeneous_coordinates.reshape(-1, 4).T
        transformed_coordinates = transformed_coordinates.T.reshape(*shape, 4)
        transformed_coordinates = transformed_coordinates[..., :3] - coordinates
        
        return transformed_coordinates
    else:
        
        field = nib.load(ants_file)
        field = field.get_fdata()
        field = np.squeeze(field)
        
        return field

def ANTs_transformations_compose(ants_files, shape):
    dvf = torch.zeros((1, 3, *shape), dtype=torch.float32)
    for ants_file in ants_files:
        dvf_t = ANTs_transformation_to_dvf(ants_file, shape=shape)
        dvf_t = torch.from_numpy(dvf_t).permute(3,0,1,2).unsqueeze(0).float()
        dvf = dvf + SpatialTransformer()(dvf_t, dvf)
    return dvf.squeeze().permute(1,2,3,0).numpy()



if __name__ == "__main__":
    pa = r"/mnt/18TB_HDD2/phz/488-647-soma-axon/to_reg/P0_Brain1/250305_P0-Brain1-TH-647-Ri-030524_16-38-48.tiff"
    pb = r"/mnt/18TB_HDD2/phz/488-647-soma-axon/to_reg/P0_Brain1/250306_P0-Brain1-TH-488-Ri-030524_17-42-21.tiff"
    a = sitk.GetArrayFromImage(sitk.ReadImage(pa))
    b = sitk.GetArrayFromImage(sitk.ReadImage(pb))

    mytx = ants.registration(fixed=ants.from_numpy(a), moving=ants.from_numpy(b), type_of_transform="Affine")
    ANTS_Tool.save_ants_params(mytx, r"/media/root/HDD3/phz/utils/test")
    wb, = ANTS_Tool.reg_with_params(r"/media/root/HDD3/phz/utils/test", [ants.from_numpy(b)])
    sitk.WriteImage(sitk.GetImageFromArray(wb.numpy()), r"/media/root/HDD3/phz/utils/test/b.nii.gz")

