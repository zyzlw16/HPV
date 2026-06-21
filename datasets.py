import os
import json
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import numpy as np
import random
from torchvision.transforms import functional as TF
import nibabel as nib  # 用于加载 NIFTI 文件格式
from scipy.ndimage import rotate  # 用于旋转操作

class HPVDataLoader(Dataset):
    """
        自定义数据加载器，用于加载 HPV 数据，包括图像和对应的标签。
        支持 NIFTI 文件的加载和数据增强。
    """
    def __init__(self, json_path, data_key, augmentations=None):
        with open(json_path, 'r') as f:
            self.data = json.load(f)[data_key]
        self.augmentations = augmentations

    def __getitem__(self, index):
        """
                根据索引返回一个样本，包括图像、mask 和标签。

                Args:
                    index (int): 数据索引。

                Returns:
                    torch.Tensor: 图像张量，形状为 (C, D,H, W)。
                    torch.Tensor: 标签张量，形状为标量。
                    str: 数据 ID（唯一标识符）。
        """
        # 获取当前样本
        item = self.data[index]

        # 加载图像和 mask
        img = self._load_nifti(item["img_path"])
        mask = self._load_nifti(item["mask_path"])
        label = item["label"]
        # 获取患者 ID
        patient_id = item["patient_id"]

        # 两通道输入
        x = np.stack([img, mask], axis=0)

        # 执行数据增强（如果有）
        if self.augmentations:
            x = self._apply_augmentations(x)

        # 转换为 PyTorch 张量
        return torch.tensor(x, dtype=torch.float32), torch.tensor(label, dtype=torch.float32), patient_id

    def __len__(self):
        return len(self.data)

    def _load_nifti(self, path):
        # 加载 NIFTI 文件
        nifti_img = nib.load(path)
        img_data = nifti_img.get_fdata()
        return img_data  # 示例，实际可能需要 nibabel 库

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


class BalancedSampler(Sampler):
    def __init__(self, labels):
        self.labels = labels
        self.class_counts = np.bincount(labels)
        self.minority_class = np.argmin(self.class_counts)
        self.majority_class = np.argmax(self.class_counts)

    def __iter__(self):
        minority_indices = [i for i, label in enumerate(self.labels) if label == self.minority_class]
        majority_indices = [i for i, label in enumerate(self.labels) if label == self.majority_class]

        sampled_minority = np.random.choice(minority_indices, size=len(majority_indices), replace=True)
        sampled_majority = np.random.choice(majority_indices, size=len(majority_indices), replace=False)

        all_indices = np.concatenate([sampled_minority, sampled_majority])
        np.random.shuffle(all_indices)
        return iter(all_indices)

    def __len__(self):
        return len(self.labels)


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

        # 数据增强（如果需要）
        if self.augmentations:
            x1 = self._apply_augmentations(x1)
            x2 = self._apply_augmentations(x2)

        # 转换为 Tensor 并返回
        gtvp_input = torch.tensor(x1, dtype=torch.float32)  # 双通道
        gtvn_input = torch.tensor(x2, dtype=torch.float32)  # 双通道
        label = torch.tensor(label, dtype=torch.long)

        return gtvp_input, gtvn_input, label

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
