from pathlib import Path

import torch
import matplotlib.pyplot as plt
from PIL import Image

import numpy as np

from tqdm import tqdm as tqdm

import albumentations as albu
import cv2
from skimage import morphology

def get_file_label(filename, args):
    filename = str(filename)
    label_str = filename.split(']')[0].split('[')[-1]
    if args.dataset == 'luad':
        l = [int(label_str[0]),int(label_str[2]),int(label_str[4]),int(label_str[6])]
    elif args.dataset == 'bcss':
        l = [int(label_str[0]),int(label_str[1]),int(label_str[2]),int(label_str[3])]
    elif args.dataset == 'wsss4luad':
        l = [int(label_str[0]), int(label_str[3]), int(label_str[6])]
    return np.array(l)

def get_label(image_path):
    """
    从图片路径中分割出字符串类型的标签
    例如，将路径
       training/1003370-11223-11698-[1, 1, 0].png
    变为
       "[1, 1, 0]"
    """
    if isinstance(image_path, Path):
        image_path = str(image_path)
    return image_path.split("-")[-1].split(".")[0]


def to_list(label_str):
    """
    把字符串形式的标签转化为list
    例如，将“[1, 0, 0]”转化为[1, 0, 0]
    :param label_str: 字符串形式的标签
    :return: 标签对应的list
    """
    label = label_str[1:-1].split(", ")

    for i in range(len(label)):
        label[i] = int(label[i])

    return label

def get_title(image_path, shape):
    # 将标签分割出来
    label = get_label(image_path)
    # 将标签转为list
    label = str(to_list(label))

    shape = str(shape)

    return "label: " + label + ", shape: " + shape

def is_tumor(image):
    # image是Path对象
    if isinstance(image, Path):
        image = str(image)
    # image是图片路径
    if isinstance(image, str):
        image = to_list(get_label(image))
    return image[0] == 1

# 判断图片是否包含肿瘤间质
def is_stroma(image):
    # image是Path对象
    if isinstance(image, Path):
        image = str(image)
    # image是图片路径
    if isinstance(image, str):
        image = to_list(get_label(image))
    return image[1] == 1

# 判断图片是否包含正常细胞
def is_normal(image):
    # image是Path对象
    if isinstance(image, Path):
        image = str(image)
    # image是图片路径
    if isinstance(image, str):
        image = to_list(get_label(image))
    return image[2] == 1

def visualize(**images):
    """PLot images in one row."""
    fontsize=14
    n = len(images)
    fig, axarr = plt.subplots(nrows=1, ncols=n, figsize=(8, 8))
    for i, (name, image) in enumerate(images.items()):
        if isinstance(image, torch.Tensor):
            if image.ndim == 3: image = image.permute(1, 2, 0)
            if image.is_cuda: image = image.detach().cpu().numpy()
        if 'mask' in name: 
            palette = [0, 64, 128, 64, 128, 0, 243, 152, 0, 255, 255, 255] + [0] * 252 * 3
            image = Image.fromarray(np.uint8(image), mode='P')
            image.putpalette(palette)
            axarr[i].imshow(image)
            axarr[i].set_title(name, fontsize=fontsize)
        else:
            axarr[i].imshow(image)
            axarr[i].set_title(name, fontsize=fontsize)
            
    for ax in axarr.ravel():
        ax.set_yticks([])
        ax.set_xticks([])
    plt.tight_layout()
    # plt.show()
    # plt.close()
    return fig

def crop_transform(height=128, width=128):
    _transform = [
        albu.RandomCrop(width=width, height=height)
    ]
    return albu.Compose(_transform)

def concat_tile(im_list_2d):
    temp = []
    for im_list_h in im_list_2d:
        hconcat = np.hstack(im_list_h)
        temp.append(hconcat)
    vconcat = np.vstack(temp)
    return vconcat

def _get_mask_and_result(image, l_bound, r_bound):
    """
    根据颜色边界l_bound和r_bound获取图像中颜色处于范围内的像素点的mask和图像
    @params:
        image:
            待处理图像（任意图像空间）
        l_bound:
            颜色下边界
        r_bound:
            颜色上边界
    @return:
        mask:
            分割出的像素点位置，使用0-255矩阵表示
        result:
            分割出的图像结果（原图像空间中）

    """
    mask = cv2.inRange(image, l_bound, r_bound)
    result = cv2.bitwise_and(image, image, mask=mask)

    return mask, result

def get_background(region):
    gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
    ret, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    binary = np.uint8(binary)    
    dst = morphology.remove_small_objects(binary==255,min_size=50,connectivity=1)
    mask = np.asarray(dst, dtype=np.uint8)
    mask = mask * 255

    return mask
    

class PolyOptimizer(torch.optim.SGD):

    def __init__(self, params, lr, weight_decay, max_step, momentum=0.9):
        super().__init__(params, lr, weight_decay)

        self.global_step = 0
        self.max_step = max_step
        self.momentum = momentum

        self.__initial_lr = [group['lr'] for group in self.param_groups]


    def step(self, closure=None):

        if self.global_step < self.max_step:
            lr_mult = (1 - self.global_step / self.max_step) ** self.momentum

            for i in range(len(self.param_groups)):
                self.param_groups[i]['lr'] = self.__initial_lr[i] * lr_mult

        super().step(closure)

        self.global_step += 1