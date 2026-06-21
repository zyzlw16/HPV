import json
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import numpy as np
import random
import nibabel as nib  # 用于加载 NIFTI 文件格式
from scipy.ndimage import rotate  # 用于旋转操作

class DualRegionDataLoader(Dataset):
    """
    新的数据加载器：支持加载 GTVp 和 GTVn 双区域的图像和对应的标签。
    """

    def __init__(self, json_path, data_key, augmentations=None):
        with open(json_path, 'r') as f:
            self.data = json.load(f)[data_key]
        self.augmentations = augmentations

    def __getitem__(self, index):
        item = self.data[index]

        # 加载 GTVp 图像和 mask
        gtvp_img = nib.load(item["GTVp_img_path"]).get_fdata()
        gtvp_mask = nib.load(item["GTVp_mask_path"]).get_fdata()
        # 两通道输入
        x1 = np.stack([gtvp_img, gtvp_mask], axis=0)

        # 加载 GTVn 图像和 mask
        gtvn_img = nib.load(item["GTVn_img_path"]).get_fdata()
        gtvn_mask = nib.load(item["GTVn_mask_path"]).get_fdata()
        x2 = np.stack([gtvn_img, gtvn_mask], axis=0)

        # 标签
        label = item["label"]
        # 获取患者 ID
        patient_id = item["patient_id"]

        # 数据增强（如果需要）
        if self.augmentations:
            x1 = self._apply_augmentations(x1)
            x2 = self._apply_augmentations(x2)

        # 转换为 Tensor 并返回
        gtvp_input = torch.tensor(x1, dtype=torch.float32)  # 双通道
        gtvn_input = torch.tensor(x2, dtype=torch.float32)  # 双通道
        label = torch.tensor(label, dtype=torch.long)

        return gtvp_input, gtvn_input, label,patient_id

    def __len__(self):
        return len(self.data)


    def _apply_augmentations(self,x):
        """
        应用 3D 数据增强操作（50% 概率随机选择一种增强方式）。

        Args:
            x (np.ndarray): 输入的 3D 图像数据，形状为 (C, D, H, W)。
            augmentations (dict): 数据增强配置，包含以下选项：
                - "flip": 是否启用随机翻转。
                - "rotate": 是否启用随机旋转。
                - "transpose": 是否启用随机转置。

        Returns:
            np.ndarray: 数据增强后的 3D 图像数据。
        """
        # 50% 概率进行增强
        if random.random() > 0.5:
            return x  # 不做增强，直接返回原始数据

        # 随机选择一种增强方式
        augmentation_type = random.choice(["flip", "rotate", "transpose"])

        if augmentation_type == "flip" and self.augmentations.get("flip", False):
            # 随机选择一个翻转方向：水平 (W)，垂直 (H)，或深度 (D)
            flip_axis = random.choice([1, 2, 3])  # 1=D, 2=H, 3=W
            x = np.flip(x, axis=flip_axis)

        elif augmentation_type == "rotate" and self.augmentations.get("rotate", False):
            # 在 xy 平面内随机旋转一个角度（-15 到 15 度）
            angle = random.uniform(-15, 15)  # 随机角度
            x = rotate(x, angle, axes=(2, 3), reshape=False, order=1, mode='nearest')

        elif augmentation_type == "transpose" and self.augmentations.get("transpose", False):
            # 随机选择一个方向进行转置
            transpose_axes = random.choice([(0, 1, 3, 2), (0, 2, 1, 3), (0, 3, 2, 1)])  # 三种可能的转置方式
            x = np.transpose(x, axes=transpose_axes)

        return x.copy()  # 确保返回的是连续的数组