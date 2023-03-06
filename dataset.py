from pathlib import Path
import numpy as np

from PIL import Image
from torch.utils.data import Dataset as BaseDataset
from tqdm import tqdm as tqdm
import cv2

import torch
import random

import albumentations as albu
from albumentations.pytorch.transforms import ToTensorV2

import utils

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from skimage import morphology

def create_data(train_data):
    print(train_data)
    tumor_set, stroma_set, normal_set = set(), set(), set()
    for path in Path(train_data).glob('*.png'):
        if utils.is_tumor(path): tumor_set.add(str(path))
        if utils.is_stroma(path): stroma_set.add(str(path))
        if utils.is_normal(path): normal_set.add(str(path))

    tumor_images = list(tumor_set - stroma_set - normal_set)
    stroma_images = list(stroma_set - tumor_set - normal_set)
    normal_images = list(normal_set - tumor_set - stroma_set)

    return tumor_images, stroma_images, normal_images

class MosaicDataset(BaseDataset):
    def __init__(self, args):
        self.args = args
        self.mosaic_data = Path(args.mosaic_data)
        self.mosaic_image = sorted(list((self.mosaic_data / 'img').glob('*.png')))
        
        self.transforms = albu.Compose([
            albu.RandomResizedCrop(height=args.patch_size, width=args.patch_size, scale=(0.9, 1)),
            albu.HorizontalFlip(p=0.5),
            albu.VerticalFlip(p=0.5),
            albu.ShiftScaleRotate(p=0.5),
            albu.OpticalDistortion(p=0.5),
            albu.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),  # Normalization
            ToTensorV2(transpose_mask=True),  # [H, W, C] -> [C, H, W]
        ])
        self.printed = False

    def __getitem__(self, i):
        H = W = self.args.patch_size
        
        name = self.mosaic_image[i].name
        image = np.array(Image.open(self.mosaic_image[i]))
        mask = np.array(Image.open(self.mosaic_data / 'mask' / name))
        
        sample = self.transforms(image=image, mask=mask)
        image, mask = sample['image'], sample['mask'].long()

        return {'image': image, 'mask': mask}
    
    def __len__(self):
        return len(self.mosaic_image)

class TrainDataset(BaseDataset):
    def __init__(self, args):
        self.args = args

        train_data = Path(args.train_data)
        self.train_image = sorted(list(train_data.glob('*.png')))

        self.transforms = albu.Compose([
            albu.Resize(args.patch_size, args.patch_size),
            albu.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            ToTensorV2(transpose_mask=True),
        ])

    def __getitem__(self, i):
        name = self.train_image[i].name
        image = np.array(Image.open(self.train_image[i]))

        if self.args.dataset == 'wsss4luad':
            background = self.__class__._get_background(image)
            tissue = np.zeros((image.shape[0], image.shape[1]))
            tissue[background == 255] = 0
            tissue[background == 0] = 127
        else:
            tissue = np.ones((image.shape[0], image.shape[1])) * 127

        sample = self.transforms(image=image, mask=tissue)
        image, tissue = sample['image'], sample['mask']
        
        return {'image': image, 'tissue': tissue, 'name': str(name)}

    def __len__(self):
        return len(self.train_image)

    @staticmethod
    def _get_background(region):
        gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
        ret, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        binary = np.uint8(binary)    
        dst = morphology.remove_small_objects(binary==255,min_size=50,connectivity=1)
        mask = np.array(dst, dtype=np.uint8)
        mask = mask * 255

        return mask

# ----> Stage 3 dataset: For second time training
class PseudoTrainDataset(BaseDataset):
    def __init__(self, args, mask_dir):
        self.args = args
        self.mask_dir = Path(mask_dir)
        self.train_dir = Path(args.train_data)
        # self.normal_images = self.__class__.__get_normal_images(self.train_dir)
        self.train_mask = sorted(list(self.mask_dir.glob('*.png')))
        self.transforms = albu.Compose([
            albu.RandomResizedCrop(height=args.patch_size, width=args.patch_size, scale=(0.9, 1)),
            albu.HorizontalFlip(p=0.5),
            albu.VerticalFlip(p=0.5),
            albu.ShiftScaleRotate(p=0.5),
            albu.OpticalDistortion(p=0.5),
            albu.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),  # Normalization
            ToTensorV2(transpose_mask=True),  # [H, W, C] -> [C, H, W]
        ])
    
    @staticmethod
    def __get_normal_images(train_dir):
        normal_set = set()

        for path in Path(train_dir).glob('*.png'):
            if utils.is_normal(path) and \
                not utils.is_tumor(path) and not utils.is_stroma(path): 
                normal_set.add(str(path))

        normal_images = sorted(list(normal_set))

        return normal_images

    @staticmethod
    def _sampling(p):
        return True if random.random() < p else False

    def __getitem__(self, i):
        # if i < len(self.train_mask):
        name = self.train_mask[i].name
        image = np.array(Image.open(self.train_dir / name))
        mask = np.array(Image.open(self.mask_dir / name))
        if self.args.dataset == 'wss4luad':
            label = np.array(utils.to_list(utils.get_label(self.train_mask[i])))
        else:
            label = utils.get_file_label(self.train_mask[i], self.args)
        
        if self.args.cutmix_pseudo and np.random.rand() < self.args.cutmix_prob:
            
            choice = np.random.choice(self.train_mask)
            mix_name = choice.name
            mix_label = np.array(utils.to_list(utils.get_label(choice)))
            mix_image = np.array(Image.open(self.train_dir / mix_name))
            mix_mask = np.array(Image.open(self.mask_dir / mix_name))

            lam = np.random.beta(1, 1)

            H, W = min(image.shape[0], mix_image.shape[0]), min(image.shape[1], mix_image.shape[1])

            bbx1, bbx2, bby1, bby2 = self._get_cutmix_bbox(W, H, lam)
            image[bbx1: bbx2, bby1: bby2, :] = mix_image[bbx1:bbx2, bby1:bby2, :]
            mask[bbx1: bbx2, bby1: bby2] = mix_mask[bbx1:bbx2, bby1:bby2]
            label = lam * label + (1 - lam) * mix_label
            
        sample = self.transforms(image=image, mask=mask)
        image, mask = sample['image'], sample['mask'].long()
        return {'image': image, 'mask': mask, 'label': torch.Tensor(label)}
    
    def _get_cutmix_bbox(self, W, H, lam):
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        cx = np.random.randint(H)
        cy = np.random.randint(W)

        bbx1 = np.clip(cx - cut_h // 2, 0, H)
        bby1 = np.clip(cy - cut_w // 2, 0, W)
        bbx2 = np.clip(cx + cut_h // 2, 0, H)
        bby2 = np.clip(cy + cut_w // 2, 0, W)
        
        return bbx1, bbx2, bby1, bby2

    def __len__(self):
        return len(self.train_mask)


class CutMixDataset(BaseDataset):
    def __init__(self, args):
        self.args = args
        train_images = sorted(list(Path(args.train_data).glob('*.png')))
        self.tumor_image, self.stroma_image, self.normal_image = self._get_one_label_set(train_images)
        self.one_label_dataset = sorted(list(self.tumor_image | self.stroma_image | self.normal_image))
        self.transforms = albu.Compose([
            albu.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            ToTensorV2(transpose_mask=True),
        ])
        
    def __getitem__(self, i):
        H = W = self.args.patch_size
        image_path = self.one_label_dataset[i]
        image = np.array(Image.open(image_path).resize((W, H), Image.BICUBIC))
        label = utils.to_list(utils.get_label(image_path))
        mask = np.full((self.args.patch_size, self.args.patch_size), label.index(1))

        background = utils.get_background(image)
        mask[background == 255] = 3
            
        if np.random.rand() < self.args.cutmix_prob:
            if image_path in self.tumor_image:
                choice_from = list(self.stroma_image | self.normal_image)
            elif image_path in self.stroma_image:
                choice_from = list(self.tumor_image | self.normal_image)
            elif image_path in self.normal_image:
                choice_from = list(self.tumor_image | self.stroma_image)
            choice = np.random.choice(choice_from)
            mix_image = np.array(Image.open(choice).resize((W, H), Image.BICUBIC))
            mix_label = utils.to_list(utils.get_label(choice))
            mix_mask = np.full((self.args.patch_size, self.args.patch_size), mix_label.index(1))
            
            background = utils.get_background(mix_image)
            mix_mask[background == 255] = 3
            
            lam = np.random.beta(1, 1)
            bbx1, bbx2, bby1, bby2 = self._get_cutmix_bbox(W, H, lam)
            image[bbx1: bbx2, bby1: bby2, :] = mix_image[bbx1:bbx2, bby1:bby2, :]
            mask[bbx1: bbx2, bby1: bby2] = mix_mask[bbx1:bbx2, bby1:bby2]
            
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (H * W))
            label = np.array(label) * lam + np.array(mix_label) * (1. - lam)
        sample = self.transforms(image=image, mask=mask)
        image, mask = sample['image'], sample['mask']
        return {'image': image, 'mask': mask}
    
    def _get_one_label_set(self, train_images):
        tumor_image, stroma_image, normal_image = set(), set(), set()
        for image in train_images:
            label = utils.to_list(utils.get_label(image))
            if sum(label) == 1:
                if label.index(1) == 0: tumor_image.add(image)
                elif label.index(1) == 1: stroma_image.add(image)
                elif label.index(1) == 2: normal_image.add(image)
                else: raise Exception
        return tumor_image, stroma_image, normal_image
    
    def _get_cutmix_bbox(self, W, H, lam):
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        cx = np.random.randint(H)
        cy = np.random.randint(W)

        bbx1 = np.clip(cx - cut_h // 2, 0, H)
        bby1 = np.clip(cy - cut_w // 2, 0, W)
        bbx2 = np.clip(cx + cut_h // 2, 0, H)
        bby2 = np.clip(cy + cut_w // 2, 0, W)
        
        return bbx1, bbx2, bby1, bby2
    
    def __len__(self):
        return len(self.one_label_dataset)
        
class TestDataset(BaseDataset):
    def __init__(self, args):
        self.args = args
        self.test_data = Path(args.test_data)
        self.test_image = sorted(list((self.test_data / 'img').glob('*.png')))
        self.transforms = albu.Compose([
            albu.PadIfNeeded(self.args.patch_size, self.args.patch_size, border_mode=2, position=albu.PadIfNeeded.PositionType.TOP_LEFT),
            albu.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            ToTensorV2(transpose_mask=True),
        ])


    def __getitem__(self, i):
        name = self.test_image[i].name
        image = self.test_image[i]
        image = np.array(Image.open(image))
        mask = np.array(Image.open(self.test_data / 'mask' / name))
        original_h, original_w = image.shape[:2]

        sample = self.transforms(image=image, mask=mask)
        image, mask = sample['image'], sample['mask'].long()

        return image, mask, name, original_h, original_w

    def __len__(self):
        return len(self.test_image)

class ValidationDataset(BaseDataset):
    def __init__(self, args, val_data):
        self.args = args
        self.val_data = Path(val_data)
        self.val_image = sorted(list((self.val_data / 'img').glob('*.png')))
        self.transforms = albu.Compose([
            albu.PadIfNeeded(self.args.patch_size, self.args.patch_size, border_mode=2, position=albu.PadIfNeeded.PositionType.TOP_LEFT),
            albu.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),  # Normalization
            ToTensorV2(transpose_mask=True),  # [H, W, C] -> [C, H, W]
        ])

    def __getitem__(self, i):
        image = self.val_image[i]
        name = image.name
        mask = self.val_data / 'mask' / name

        image = np.array(Image.open(image))  # mode='RGB'
        mask = np.array(Image.open(mask))  # [H, W, C]
        original_h, original_w = image.shape[:2]

        sample = self.transforms(image=image, mask=mask)
        image, mask = sample['image'], sample['mask'].long()

        return image, mask, name, original_h, original_w

    def __len__(self):
        return len(self.val_image)