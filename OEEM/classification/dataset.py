from pathlib import Path
import torch
import re
from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image
from utils.pyutils import multiscale_online_crop
from torchvision import transforms

def get_file_label(filename, num_class=2):
    if num_class == 4: # bcss
        label_str = filename.split('[')[-1].split(']')[0]
        l = [label_str[0], label_str[1], label_str[2], label_str[3]]
        l = list(map(lambda x: int(x), l))
    else:
        label_str = '.'.join(filename.split('.')[:-1]).split('-')[-1]
        l = eval(label_str)
    assert len(l) == num_class
    return np.array(l)

class OriginPatchesDataset(Dataset):
    def __init__(self, data_path_name=None, transform=None, num_class=2):
        self.path = data_path_name
        self.files = [i.name for i in sorted(list(Path(data_path_name).glob('*.png')))]
        self.transform = transform
        self.filedic = {}
        self.num_class = num_class

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.path, self.files[idx])
        im = Image.open(image_path)
        label = get_file_label(filename=self.files[idx], num_class=self.num_class)
        im = transforms.ToTensor()(im)

        if self.transform:
            im = self.transform(im)
        
        return im, label

class OfflineDataset(Dataset):
    def __init__(self, dataset_path, transform):
        self.path = dataset_path
        self.files = os.listdir(self.path)
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.path, self.files[idx])
        im = Image.open(image_path)
        positions = self.files[idx]
        positions = list(map(lambda x: int(x), re.findall(r'\d+', positions)))
        if self.transform:
            im = self.transform(im)
        return im, np.array(positions)

class TrainingSetCAM(Dataset):
    def __init__(self, data_path_name, transform, patch_size, stride, scales, num_class=2):
        self.path = data_path_name
        self.files = [i.name for i in sorted(list(Path(data_path_name).glob('*.png')))]
        self.transform = transform
        self.patch_size = patch_size
        self.stride = stride
        self.scales = scales
        self.num_class = num_class

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.path, self.files[idx])
        im = np.array(Image.open(image_path))
        scaled_im_list, scaled_position_list = multiscale_online_crop(im, self.patch_size, self.stride, self.scales)
        if self.transform:
            for im_list in scaled_im_list:
                for patch_id in range(len(im_list)):
                    im_list[patch_id] = self.transform(im_list[patch_id])

        label = get_file_label(image_path, num_class=self.num_class)

        return self.files[idx], scaled_im_list, scaled_position_list, self.scales, label
