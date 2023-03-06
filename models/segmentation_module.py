from PIL import Image
from einops import rearrange, repeat
# import monai
import numpy as np
from pathlib import Path
from functools import partial
import os

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import ttach as tta

import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss, FocalLoss
import torch.nn as nn

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.utils import save_image
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import ExponentialLR

import pytorch_lightning as pl

import utils
from utils import get_label, to_list
from loss import mIoUMask

import logging

class UnNormalize(object):
    def __init__(self, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

def mask_to_numpy(mask):
    palette = [0, 64, 128, 64, 128, 0, 243, 152, 0, 255, 255, 255] + [0] * 252 * 3
    image = Image.fromarray(np.uint8(mask), mode='P')
    image.putpalette(palette)
    image = image.convert('RGB')
    image = np.asarray(image)
    image = rearrange(image, 'h w c -> c h w')
    return image

def unormalize(image):
    return UnNormalize()(image)

class SegmentationModule(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.retrain_iteration = 0

        self.patch_size = args.patch_size
        
        # self.train_dice = DiceLoss(mode='multiclass', ignore_index=3)
        if self.args.dataset == 'wsss4luad':
            self.train_ce = nn.CrossEntropyLoss(reduction='none', ignore_index=3)
        else:
            self.train_ce = nn.CrossEntropyLoss(reduction='none')

        self.train_iou = mIoUMask(num_classes=args.num_classes)
        self.valid_iou = mIoUMask(num_classes=args.num_classes)
        self.test_iou = mIoUMask(num_classes=args.num_classes)
        
        if 'Unet' in args.model:
            self.model = smp.create_model(
                self.args.model, encoder_name=self.args.encoder, in_channels=3, classes=self.args.num_classes, 
                decoder_attention_type='scse',
                # aux_params=aux_params,
            )
        else:
            self.model = smp.create_model(
                self.args.model, encoder_name=self.args.encoder, in_channels=3, classes=self.args.num_classes, 
            )


        self.save_hyperparameters()
    
    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), self.args.lr, weight_decay=self.args.weight_decay)
        scheduler = ExponentialLR(optimizer, gamma=0.9)

        return [optimizer], [scheduler]

    def forward(self, x):
        y = self.model(x)
        return y
    
    def training_step(self, batch, batch_idx): 
        result = self.model(batch['image'])
        mask_pred = result
        mask, label = batch['mask'], batch['label']
        # dice = self.train_dice(mask_pred, mask)
        ce = self.train_ce(mask_pred, mask)
        loss = torch.mean(ce)
        # loss = dice + ce_mean
        # loss = self.train_dice(mask_pred, mask)  
        
        self.log('train_loss', loss, prog_bar=True)

        self.train_iou(mask_pred, mask)
        self.log('train_miou', self.train_iou.Mean_Intersection_over_Union(), prog_bar=True)
        
        return loss
    
    def training_epoch_end(self, training_step_outputs):
        self.log('train_miou_epoch', self.train_iou.Mean_Intersection_over_Union())
        self.log('train_fwiou_epoch', self.train_iou.Frequency_Weighted_Intersection_over_Union())
        self.train_iou.reset()

    def on_validation_epoch_start(self):
        if self.args.dataset == 'wsss4luad':
            self.pred_big_mask_dict_ms = dict()
            self.cnt_big_mask_dict_ms = dict()
            self.pred_big_mask_dict = dict()
            self.cnt_big_mask_dict = dict()
        else:
            pass

    def validation_step(self, batch, batch_idx):
        image_batch, mask_batch, name_batch, original_h_batch, original_w_batch = batch
        output = self(image_batch)
       
        self.valid_iou(output, mask_batch)

        if self.args.dataset == 'wsss4luad':
            # ----> Process each sample
            for j in range(image_batch.shape[0]):
                original_w, original_h = original_w_batch[j], original_h_batch[j]

                output_ = output[j][:, :original_h, :original_w] # logits, C x 256 x 256
                
                probs = torch.softmax(output_, dim=0).cpu().numpy()
                probs = probs.transpose(1, 2, 0) # [H, W, C]

                name = name_batch[j]
                image_idx = name.split('_')[0]
                scale = float(name.split('_')[1])
                position = (int(name.split('_')[2]), int(name.split('_')[3].split('-')[0]))

                dict_key = f'{image_idx}_{scale}'

                if dict_key not in self.pred_big_mask_dict_ms:
                    w, h = Image.open(os.path.join('/'.join(self.args.val_data.split('/')[:-1]), 'img', image_idx + '.png')).size
                    w_ = int(w * scale)
                    h_ = int(h * scale)
                    self.pred_big_mask_dict_ms[dict_key] = np.zeros((h_, w_, 3))
                    self.cnt_big_mask_dict_ms[dict_key] = np.zeros((h_, w_, 1))
                    
                self.pred_big_mask_dict_ms[dict_key][position[0]:position[0]+output_.shape[1], position[1]:position[1]+output_.shape[2], :] += probs
                self.cnt_big_mask_dict_ms[dict_key][position[0]:position[0]+output_.shape[1], position[1]:position[1]+output_.shape[2], :] += 1


    def validation_epoch_end(self, validation_step_outputs):
        if self.args.dataset == 'wsss4luad':
            # ----> All validation patches are predicted, now we can calculate the final miou
            for k, mask in self.pred_big_mask_dict_ms.items():
                mask /= self.cnt_big_mask_dict_ms[k] # [H, W, 3]
                image_idx = k.split('_')[0]

                if image_idx not in self.pred_big_mask_dict:
                    w, h = Image.open(os.path.join('/'.join(self.args.val_data.split('/')[:-1]), 'img', image_idx + '.png')).size
                    self.pred_big_mask_dict[image_idx] = np.zeros((h, w, 3))
                    self.cnt_big_mask_dict[image_idx] = np.zeros((h, w, 1))

                mask = F.interpolate(torch.from_numpy(mask.transpose(2, 0, 1)).unsqueeze(0), (h, w), mode='bilinear')[0].numpy().transpose(1, 2, 0)
                self.pred_big_mask_dict[image_idx][:, :, :] += mask
                self.cnt_big_mask_dict[image_idx][:, :, :] += 1

            big_mask_iou = mIoUMask()
            for k, mask_pred in self.pred_big_mask_dict.items():
                mask_pred /= self.cnt_big_mask_dict[k]
                mask = np.asarray(Image.open(os.path.join('/'.join(self.args.val_data.split('/')[:-1]), 'mask', k + '.png')))

                big_mask_iou(torch.from_numpy(mask_pred.transpose(2, 0, 1)).unsqueeze(0), torch.from_numpy(mask).unsqueeze(0), probs=True)

            # ----> Print and Log metrics
            tissue_iou = self.valid_iou.Tissue_Intersection_over_Union()
            print('\n' + '-' * 50)
            print("\nExperiment Settings")
            print(f"Dataset: \033[1;34m{self.args.pseudo_mask_dir}\033[0m")
            print(f"Log Path: \033[1;34m{self.args.log_path}\033[0m")

            print('\n' + '-' * 50)
            print(f"\nValidation Result (Patch)")
            print(f'Tumor IoU: \033[1;35m{tissue_iou[0]:.4f}\033[0m')
            print(f'Stroma IoU: \033[1;35m{tissue_iou[1]:.4f}\033[0m')
            print(f'Normal IoU: \033[1;35m{tissue_iou[2]:.4f}\033[0m')
            print(f'mIoU: \033[1;35m{self.valid_iou.Mean_Intersection_over_Union():.4f}\033[0m')
            print(f'fwIoU: \033[1;35m{self.valid_iou.Frequency_Weighted_Intersection_over_Union():.4f}\033[0m')
            print('\n' + '-' * 50)

            self.log(f'validation_tiou_patch_epoch', tissue_iou[0], prog_bar=False)
            self.log(f'validation_siou_patch_epoch', tissue_iou[1], prog_bar=False)
            self.log(f'validation_niou_patch_epoch', tissue_iou[2], prog_bar=False)
            self.log(f'validation_miou_patch_epoch', self.valid_iou.Mean_Intersection_over_Union(), prog_bar=True)
            self.log(f'validation_fwiou_patch_epoch', self.valid_iou.Frequency_Weighted_Intersection_over_Union(), prog_bar=True)

            self.valid_iou.reset()

            big_tissue_iou = big_mask_iou.Tissue_Intersection_over_Union()
            print('\n' + '-' * 50)
            print(f"\nValidation Result (Big Mask)")
            print(f'Tumor IoU: \033[1;35m{big_tissue_iou[0]:.4f}\033[0m')
            print(f'Stroma IoU: \033[1;35m{big_tissue_iou[1]:.4f}\033[0m')
            print(f'Normal IoU: \033[1;35m{big_tissue_iou[2]:.4f}\033[0m')
            print(f'mIoU: \033[1;35m{big_mask_iou.Mean_Intersection_over_Union():.4f}\033[0m')
            print(f'fwIoU: \033[1;35m{big_mask_iou.Frequency_Weighted_Intersection_over_Union():.4f}\033[0m')
            print('\n' + '-' * 50)

            self.log(f'validation_tiou_mask_epoch', big_tissue_iou[0], prog_bar=False)
            self.log(f'validation_siou_mask_epoch', big_tissue_iou[1], prog_bar=False)
            self.log(f'validation_niou_mask_epoch', big_tissue_iou[2], prog_bar=False)
            self.log(f'validation_miou_mask_epoch', big_mask_iou.Mean_Intersection_over_Union(), prog_bar=True)
            self.log(f'validation_fwiou_mask_epoch', big_mask_iou.Frequency_Weighted_Intersection_over_Union(), prog_bar=True)

            big_mask_iou.reset()

        else:

            tissue_iou = self.valid_iou.Tissue_Intersection_over_Union()
            print('\n' + '-' * 50)
            print("\nExperiment Settings")
            print(f"Dataset: \033[1;34m{self.args.pseudo_mask_dir}\033[0m")
            print(f"Log Path: \033[1;34m{self.args.log_path}\033[0m")

            print('\n' + '-' * 50)
            print(f"\nValidation Result (Mask)")
            print(f'Tumor IoU: \033[1;35m{tissue_iou[0]:.4f}\033[0m')
            print(f'Stroma IoU: \033[1;35m{tissue_iou[1]:.4f}\033[0m')
            print(f'Lymphocytic infiltrate IoU: \033[1;35m{tissue_iou[2]:.4f}\033[0m')
            print(f'Necrosis IoU: \033[1;35m{tissue_iou[3]:.4f}\033[0m')
            print(f'mIoU: \033[1;35m{self.valid_iou.Mean_Intersection_over_Union():.4f}\033[0m')
            print(f'fwIoU: \033[1;35m{self.valid_iou.Frequency_Weighted_Intersection_over_Union():.4f}\033[0m')
            print('\n' + '-' * 50)

            self.log(f'validation_tmr_mask_epoch', tissue_iou[0], prog_bar=False)
            self.log(f'validation_str_mask_epoch', tissue_iou[1], prog_bar=False)
            self.log(f'validation_lym_mask_epoch', tissue_iou[2], prog_bar=False)
            self.log(f'validation_nec_mask_epoch', tissue_iou[3], prog_bar=False)
            self.log(f'validation_miou_mask_epoch', self.valid_iou.Mean_Intersection_over_Union(), prog_bar=True)
            self.log(f'validation_fwiou_mask_epoch', self.valid_iou.Frequency_Weighted_Intersection_over_Union(), prog_bar=True)

            self.valid_iou.reset()


    #----> remove v_num
    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    def show_image_on_tensorboard(self, title, global_step, **show_dict):
        tensorboard = self.loggers[0].experiment
        tensorboard.add_figure(
            title, 
            utils.visualize(**show_dict),
            global_step
        )