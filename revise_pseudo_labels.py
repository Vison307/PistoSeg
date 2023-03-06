import argparse
import os

from loss import mIoUMask
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8' # for reproducibility
import numpy as np
import random

import torch
import torchvision.transforms

from torch import nn
import torch.nn.functional as F
from torch import autograd

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import albumentations as albu
from albumentations.pytorch.transforms import ToTensorV2

from einops import rearrange
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

import utils

from tqdm import tqdm
from models.revise_net import Net

import time
import logging
execution_time = time.strftime("%Y%m%d%H%M%S", time.localtime())


class RefineDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, pmask_dir, cam_dir):
        self.pmask_dir = Path(pmask_dir) / 'logits_32x32'
        self.cam_dir = cam_dir
        self.train_images = sorted(list(Path(image_dir).glob('*.png')))
        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        
    @staticmethod
    def _downsample(tensor, target_shape):
        return F.interpolate(tensor.unsqueeze(0), target_shape, mode='bilinear')[0]

    def __getitem__(self, i):
        train_image_path = Path(self.train_images[i])
        name = str(train_image_path.stem)
        
        image = np.array(Image.open(train_image_path).resize((224, 224))) 
        image = self.transforms(image)
        
        pmask = torch.load(os.path.join(self.pmask_dir, name + '.pt'), map_location='cpu') # [32, 32]
        
        cam = np.load(os.path.join(self.cam_dir, name + '.npy')) # [32, 32]
        cam = torch.from_numpy(cam).to(torch.float32)

        label = utils.get_file_label(train_image_path, args)

        assert not torch.any(torch.isnan(image)), f'image: {image}'
        assert not torch.any(torch.isnan(pmask)), f'pmask: {pmask}'
        assert not torch.any(torch.isnan(cam)), f'cam: {cam}'
        
        return {'image': image, 'pmask': pmask, 'cam': cam, 'label': torch.Tensor(label)}

    def __len__(self):
        return len(self.train_images)
    
class RefineValidationDataset(torch.utils.data.Dataset):
    def __init__(self, val_dir_patch, val_pmask_dir, val_cam_dir, patch_size=224):
        self.pmask_dir = Path(val_pmask_dir) / 'logits_32x32'
        self.cam_dir = Path(val_cam_dir)
        self.mask_dir = Path(val_dir_patch) / 'mask'
        self.images = sorted(list((Path(val_dir_patch) / 'img').glob('*.png')))
        self.transforms = albu.Compose([
            albu.PadIfNeeded(patch_size, patch_size, border_mode=2, position=albu.PadIfNeeded.PositionType.TOP_LEFT),
            albu.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),  # Normalization
            ToTensorV2(transpose_mask=True),  # [H, W, C] -> [C, H, W]
        ])
        
    @staticmethod
    def _downsample(tensor, target_shape):
        return F.interpolate(tensor.unsqueeze(0), target_shape, mode='bilinear')[0]

    def __getitem__(self, i):
        image_path = Path(self.images[i])
        name = str(image_path.stem)
        
        image = np.array(Image.open(image_path)) 
        mask = np.array(Image.open(os.path.join(self.mask_dir, name + '.png')))
        original_h, original_w = image.shape[:2]

        sample = self.transforms(image=image, mask=mask)
        image, mask = sample['image'], sample['mask'].long()
        
        pmask = torch.load(os.path.join(self.pmask_dir, name + '.pt'), map_location='cpu') # [3, 32, 32]
        
        cam = np.load(os.path.join(self.cam_dir, name + '.npy')) # [32, 32]
        cam = torch.from_numpy(cam).to(torch.float32)

        label = np.array(utils.to_list(utils.get_label(str(image_path))))
        
        assert not torch.any(torch.isnan(image)), f'image: {image}'
        assert not torch.any(torch.isnan(pmask)), f'pmask: {pmask}'
        assert not torch.any(torch.isnan(cam)), f'cam: {cam}'

        return {'image': image, 'pmask': pmask, 'cam': cam, 'mask': mask, 'label': torch.Tensor(label), 'name': name, 'original_h': original_h, 'original_w': original_w}

    def __len__(self):
        return len(self.images)

def adaptive_min_pooling_loss(x):
    # This loss does not affect the highest performance, but change the optimial background score (alpha)
    n, c, h, w = x.size()
    k = h * w // 4
    x = torch.max(x, dim=1)[0]
    y = torch.topk(x.view(n,-1), k=k, dim=-1, largest=False)[0] # k smallest elements
    y = F.relu(y, inplace=False)
    loss = torch.sum(y)/(k*n)
    return loss

def max_onehot(x):
    """normalize the activation vectors of each pixel by supressing foreground non-maximum activations to zeros"""
    n, c, h, w = x.size()
    x_max = torch.max(x[:,1:,:,:], dim=1, keepdim=True)[0]
    x[:,1:,:,:][x[:,1:,:,:] != x_max] = 0
    return x

def max_norm(p, e=1e-5):
    N, C, H, W = p.size()
    # p = F.relu(p)
    max_v = torch.max(p.view(N,C,-1),dim=-1)[0].view(N,C,1,1)
    min_v = torch.min(p.view(N,C,-1),dim=-1)[0].view(N,C,1,1)
    p = (p - min_v) / (max_v - min_v + e)
    return p

def seed_all(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False 
    torch.use_deterministic_algorithms(True, warn_only=True)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def main(args):
    seed_all(args.seed)
    g = torch.Generator()
    g.manual_seed(args.seed)

    train_dataset = RefineDataset(args.train_dir, args.pmask_dir, args.cam_dir)
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, worker_init_fn=seed_worker, generator=g, num_workers=args.num_workers)

    # val_dataset = RefineValidationDataset(args.val_dir_patch, args.val_pmask_dir, args.val_cam_dir)
    # val_dataloader = torch.utils.data.DataLoader(
    #     val_dataset, 
    #     batch_size=128, 
    #     shuffle=False, 
    #     num_workers=args.num_workers
    # )
    
    model = Net(num_classes=args.n_class+1).cuda()
    max_step = len(train_dataset) // args.batch_size * args.max_epoches
    param_groups = model.get_parameter_groups()
    optimizer = utils.PolyOptimizer([
        {'params': param_groups[0], 'lr': args.lr, 'weight_decay': args.wt_dec},
        {'params': param_groups[1], 'lr': 2*args.lr, 'weight_decay': 0},
        {'params': param_groups[2], 'lr': 10*args.lr, 'weight_decay': args.wt_dec},
        {'params': param_groups[3], 'lr': 20*args.lr, 'weight_decay': 0}
    ], lr=args.lr, weight_decay=args.wt_dec, max_step=max_step)

    if args.weights[-7:] == '.params':
        import models.resnet38d
        weights_dict = models.resnet38d.convert_mxnet_to_torch(args.weights)
    else:
        weights_dict = torch.load(args.weights)

    model.load_state_dict(weights_dict, strict=False)
    model = torch.nn.DataParallel(model).cuda()
    model.train()
    
    # with autograd.detect_anomaly():
    loss_list = []
    loss_cls_list = []
    loss_rfm_list = []
    loss_ecr_list = []

    # best_model = None
    # best_score = 0
    for epoch in range(args.max_epoches):
        train_epoch(dataloader, model, optimizer, loss_list, loss_cls_list, loss_rfm_list, loss_ecr_list, epoch)
        # cam_rv_mask_miou = val_epoch(val_dataloader, model)
        # logging.critical(f'cam_rv_mask_miou: {cam_rv_mask_miou}')
        # if cam_rv_mask_miou > best_score:
        #     print(f'Cam_rv_mask_miou improved from {best_score} to {cam_rv_mask_miou}, update the best model!')
        #     logging.critical(f'Cam_rv_mask_miou improved from {best_score} to {cam_rv_mask_miou}, update the best model!')
        #     best_model = model
        #     best_score = cam_rv_mask_miou


    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)
    # print(f'Best Score: {best_score}')
    print(f"Final model saves in {os.path.join(args.save_dir, f'ResNet38-RFM.pth')}")
    # logging.critical(f'Best Score: {best_score}')
    logging.critical(f"Final model saves in {os.path.join(args.save_dir, f'ResNet38-RFM.pth')}")
    torch.save(model.state_dict(), os.path.join(args.save_dir, f'ResNet38-RFM.pth'))

    plt.plot(loss_list)
    plt.plot(loss_cls_list)
    plt.plot(loss_rfm_list)
    plt.plot(loss_ecr_list)
    plt.legend(['loss', 'loss_cls', 'loss_rfm', 'loss_ecr'])
    plt.savefig(os.path.join(args.save_dir, f'loss.png'))

def train_epoch(dataloader, model, optimizer, loss_list, loss_cls_list, loss_rfm_list, loss_ecr_list, epoch):
    ep_loss = []
    ep_loss_cls = []
    ep_loss_rfm = []
    ep_loss_ecr = []

    model.train()
    with tqdm(dataloader, total=len(dataloader)) as pbar:
        for idx, batch in enumerate(pbar):
            x = batch['image'].cuda()
            N, C, H, W = x.size()

            pmask = batch['pmask']
            pcam = batch['cam']
            n, c, h, w = pmask.size()
            pmask = torch.concat([torch.zeros((n, 1, h, w)), pmask], dim=1).cuda()
            pcam = torch.concat([torch.zeros((n, 1, h, w)), pcam], dim=1).cuda()

            label = batch['label']
            bg_score = torch.ones((n, 1))
            label = torch.cat((bg_score, label), dim=1)
            label = label.cuda(non_blocking=True).unsqueeze(2).unsqueeze(3) # [B, 4, 1, 1]

            # cam: [B, 4, 256, 256]
            # cam_rv: [B, 4, 256, 256]
            # pmask_rv: [B, 4, 256, 256]
            # pcam_rv: [B, 4, 256, 256]
            cam, cam_rv, pmask_rv, pcam_rv = model(x, pmask, pcam)

            # ----> Classification loss
            label_cam = F.adaptive_avg_pool2d(cam, (1, 1)) # [B, 4, 1, 1] --> classification prediction
            loss_rvmin = adaptive_min_pooling_loss((cam_rv*label)[:,1:,:,:])
            loss_cls = F.multilabel_soft_margin_loss(label_cam[:,1:,:,:], label[:,1:,:,:])
            loss_cls = loss_cls + loss_rvmin

            # loss_cls = F.binary_cross_entropy_with_logits(label_cam[:,1:,:,:], label[:,1:,:,:])

            # ----> Revise Mask Loss
            # pmask_rv = max_norm(pmask_rv) * label
            # pcam_rv = max_norm(pcam_rv) * label
            pmask_rv = pmask_rv * label
            pcam_rv = pcam_rv * label
            loss_rfm = torch.mean(torch.abs(pmask_rv[:,1:,:,:] - pcam_rv[:,1:,:,:]))


            ns, cs, hs, ws = cam.size()
            pmask = max_norm(pmask) * label
            pcam = max_norm(pcam) * label
            pmask[:, 0, :, :] = 1 - torch.max(pmask[:, 1:, :, :], dim=1)[0]
            pcam[:, 0, :, :] = 1 - torch.max(pcam[:, 1:, :, :], dim=1)[0]
            pmask = F.interpolate(pmask, (H, W), mode='bilinear', align_corners=True) # [2, 4, 256, 256]
            pcam = F.interpolate(pcam, (H, W), mode='bilinear', align_corners=True) # [2, 4, 256, 256]

            tensor_ecr1 = torch.abs(max_onehot(pmask.detach()) - pcam_rv) #*eq_mask
            tensor_ecr2 = torch.abs(max_onehot(pcam.detach()) - pmask_rv) #*eq_mask
            loss_ecr1 = torch.mean(torch.topk(tensor_ecr1.view(ns, -1), k=(int)(4*hs*ws*0.2), dim=-1)[0])
            loss_ecr2 = torch.mean(torch.topk(tensor_ecr2.view(ns, -1), k=(int)(4*hs*ws*0.2), dim=-1)[0])
            loss_ecr = loss_ecr1 + loss_ecr2

            l = loss_cls + loss_rfm + loss_ecr

            # l = loss_rfm + loss_ecr

            # l = loss_cls + loss_rfm

            # l = loss_rfm

            # l = loss_ecr

            ep_loss.append(l.item())
            ep_loss_cls.append(loss_cls.item())
            ep_loss_rfm.append(loss_rfm.item())
            ep_loss_ecr.append(loss_ecr.item())

            pbar.set_postfix(loss_cls=loss_cls.item(), loss_rfm=loss_rfm.item(), loss_ecr=loss_ecr.item(), loss=l.item())

            optimizer.zero_grad()
            l.backward()
            optimizer.step()

        else:
            loss_list.append(sum(ep_loss) / len(ep_loss))
            loss_cls_list.append(sum(ep_loss_cls) / len(ep_loss_cls))
            loss_rfm_list.append(sum(ep_loss_rfm) / len(ep_loss_rfm))
            loss_ecr_list.append(sum(ep_loss_ecr) / len(ep_loss_ecr))
            print(f'epoch: {epoch}, loss_cls: {loss_cls_list[-1]}, loss_rfm: {loss_rfm_list[-1]}, loss_ecr: {loss_ecr_list[-1]}, loss: {loss_list[-1]}')
            logging.critical(f'epoch: {epoch}, loss_cls: {loss_cls_list[-1]}, loss_rfm: {loss_rfm_list[-1]}, loss_ecr: {loss_ecr_list[-1]}, loss: {loss_list[-1]}')

@torch.no_grad()
def val_epoch(dataloader, model):
    pred_big_mask_dict_ms = dict()
    cnt_big_mask_dict_ms = dict()
    pred_big_mask_dict = dict()
    cnt_big_mask_dict = dict()

    cam_rv_iou = mIoUMask()
    # pmask_rv_iou = mIoUMask()
    # pcam_rv_iou = mIoUMask()

    model.eval()
    with tqdm(dataloader, total=len(dataloader)) as pbar:
        for idx, batch in enumerate(pbar):
            x = batch['image'].cuda()
            N, C, H, W = x.size()

            pmask = batch['pmask']
            pcam = batch['cam']
            mask = batch['mask']
            name_batch = batch['name']
            original_h_batch = batch['original_h']
            original_w_batch = batch['original_w']

            n, c, h, w = pmask.size()
            pmask = torch.concat([torch.zeros((n, 1, h, w)), pmask], dim=1).cuda()
            pcam = torch.concat([torch.zeros((n, 1, h, w)), pcam], dim=1).cuda()

            label = batch['label']
            bg_score = torch.ones((n, 1))
            label = torch.cat((bg_score, label), dim=1)
            label = label.cuda(non_blocking=True).unsqueeze(2).unsqueeze(3) # [B, 4, 1, 1]

            # cam: [B, 4, 256, 256]
            # cam_rv: [B, 4, 256, 256]
            # pmask_rv: [B, 4, 256, 256]
            # pcam_rv: [B, 4, 256, 256]
            _, cam_rv, pmask_rv, pcam_rv = model(x, pmask, pcam)

            pmask_rv = (pmask_rv * label)[:, 1:, :, :]
            pcam_rv = (pcam_rv * label)[:, 1:, :, :]
            cam_rv = (cam_rv * label)[:, 1:, :, :]

            cam_rv_iou(cam_rv, mask)
            # pmask_rv_iou(pmask_rv, mask)
            # pcam_rv_iou(pcam_rv, mask)

            if args.dataset == 'wsss4luad':

                output = cam_rv

                # ----> Process each sample
                for j in range(x.shape[0]):
                    original_w, original_h = original_w_batch[j], original_h_batch[j]

                    output_ = output[j][:, :original_h, :original_w] # logits, C x 256 x 256
                    
                    probs = torch.softmax(output_, dim=0).cpu().numpy()
                    probs = probs.transpose(1, 2, 0) # [H, W, C]

                    name = name_batch[j]
                    image_idx = name.split('_')[0]
                    scale = float(name.split('_')[1])
                    position = (int(name.split('_')[2]), int(name.split('_')[3].split('-')[0]))

                    dict_key = f'{image_idx}_{scale}'

                    if dict_key not in pred_big_mask_dict_ms:
                        w, h = Image.open(os.path.join(args.val_dir, 'img', image_idx + '.png')).size
                        w_ = int(w * scale)
                        h_ = int(h * scale)
                        pred_big_mask_dict_ms[dict_key] = np.zeros((h_, w_, 3))
                        cnt_big_mask_dict_ms[dict_key] = np.zeros((h_, w_, 1))
                        
                    pred_big_mask_dict_ms[dict_key][position[0]:position[0]+output_.shape[1], position[1]:position[1]+output_.shape[2], :] += probs
                    cnt_big_mask_dict_ms[dict_key][position[0]:position[0]+output_.shape[1], position[1]:position[1]+output_.shape[2], :] += 1
        
        if args.dataset == 'wsss4luad':
            # ----> All validation patches are predicted, now we can calculate the final miou
            for k, mask in pred_big_mask_dict_ms.items():
                mask /= cnt_big_mask_dict_ms[k] # [H, W, 3]
                image_idx = k.split('_')[0]

                if image_idx not in pred_big_mask_dict:
                    w, h = Image.open(os.path.join(args.val_dir, 'img', image_idx + '.png')).size
                    pred_big_mask_dict[image_idx] = np.zeros((h, w, 3))
                    cnt_big_mask_dict[image_idx] = np.zeros((h, w, 1))

                mask = F.interpolate(torch.from_numpy(mask.transpose(2, 0, 1)).unsqueeze(0), (h, w), mode='bilinear')[0].numpy().transpose(1, 2, 0)
                pred_big_mask_dict[image_idx][:, :, :] += mask
                cnt_big_mask_dict[image_idx][:, :, :] += 1

            big_mask_iou = mIoUMask()
            for k, mask_pred in pred_big_mask_dict.items():
                mask_pred /= cnt_big_mask_dict[k]
                mask = np.array(Image.open(os.path.join(args.val_dir, 'mask', k + '.png')))

                big_mask_iou(torch.from_numpy(mask_pred.transpose(2, 0, 1)).unsqueeze(0), torch.from_numpy(mask).unsqueeze(0), probs=True)

            # ----> Print and Log metrics
            tissue_iou = cam_rv_iou.Tissue_Intersection_over_Union()
            cam_rv_patch_miou = cam_rv_iou.Mean_Intersection_over_Union()
            print('\n' + '-' * 50)
            print(f"\nCam_rv Validation Result (Patch)")
            print(f'Tumor IoU: \033[1;35m{tissue_iou[0]:.4f}\033[0m')
            print(f'Stroma IoU: \033[1;35m{tissue_iou[1]:.4f}\033[0m')
            print(f'Normal IoU: \033[1;35m{tissue_iou[2]:.4f}\033[0m')
            print(f'mIoU: \033[1;35m{cam_rv_iou.Mean_Intersection_over_Union():.4f}\033[0m')
            print(f'fwIoU: \033[1;35m{cam_rv_iou.Frequency_Weighted_Intersection_over_Union():.4f}\033[0m')
            print('\n' + '-' * 50)

            cam_rv_iou.reset()

            big_tissue_iou = big_mask_iou.Tissue_Intersection_over_Union()
            cam_rv_mask_miou = big_mask_iou.Mean_Intersection_over_Union()
            print('\n' + '-' * 50)
            print(f"\nCAMrv Validation Result (Big Mask)")
            print(f'Tumor IoU: \033[1;35m{big_tissue_iou[0]:.4f}\033[0m')
            print(f'Stroma IoU: \033[1;35m{big_tissue_iou[1]:.4f}\033[0m')
            print(f'Normal IoU: \033[1;35m{big_tissue_iou[2]:.4f}\033[0m')
            print(f'mIoU: \033[1;35m{big_mask_iou.Mean_Intersection_over_Union():.4f}\033[0m')
            print(f'fwIoU: \033[1;35m{big_mask_iou.Frequency_Weighted_Intersection_over_Union():.4f}\033[0m')
            print('\n' + '-' * 50)

            big_mask_iou.reset()
            return cam_rv_mask_miou
        else:
            tissue_iou = cam_rv_iou.Tissue_Intersection_over_Union()
            cam_rv_miou = cam_rv_iou.Mean_Intersection_over_Union()
            print('\n' + '-' * 50)
            print(f"\nCam_rv Validation Result (Big Mask)")
            print(f'TUM IoU: \033[1;35m{tissue_iou[0]:.4f}\033[0m')
            print(f'STR IoU: \033[1;35m{tissue_iou[1]:.4f}\033[0m')
            print(f'LYM IoU: \033[1;35m{tissue_iou[2]:.4f}\033[0m')
            print(f'NEC IoU: \033[1;35m{tissue_iou[3]:.4f}\033[0m')
            print(f'mIoU: \033[1;35m{cam_rv_iou.Mean_Intersection_over_Union():.4f}\033[0m')
            print(f'fwIoU: \033[1;35m{cam_rv_iou.Frequency_Weighted_Intersection_over_Union():.4f}\033[0m')
            print('\n' + '-' * 50)

            cam_rv_iou.reset()
            return cam_rv_miou

def parse_args():

    # CUDA_VISIBLE_DEVICES=6, python revise_pseudo_labels.py --pmask_dir /data115_1/fzj/data/pmasks/miou=0.5782_20220726211929 --train_dir /data115_1/fzj/data/training --val_dir_patch /data115_1/fzj/data/validation/patches_256_128 --val_dir /data115_1/fzj/data/validation --max_epoches 10

    # CUDA_VISIBLE_DEVICES=1, python revise_pseudo_labels.py --pmask_dir mosaic_logs/Unet:efficientnet-b6:224:16:0.001 --cam_dir data/BCSS-WSSS/res38d_train_pseudo_mask/npy --train_dir data/BCSS-WSSS/training --max_epoches 25 --lr 1e-3 --save_dir mosaic_logs --dataset bcss --n_class 4

    parser = argparse.ArgumentParser(description='Infer Pseudo-Labels for dataset')

    parser.add_argument('--dataset', default='wsss4luad')
    parser.add_argument('--n_class', type=int, default=3)

    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--wt_dec', type=float, default=5e-4)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_epoches', type=int, default=30)

    parser.add_argument('--save_dir', type=str, default='./weights')
    parser.add_argument('--weights', default='./weights/ilsvrc-cls_rna-a1_cls1000_ep-0001.params')

    parser.add_argument('--pmask_dir')
    parser.add_argument('--cam_dir', default='./pmask_cam_ds/cam')
    parser.add_argument('--train_dir', default='./data/training')
    # parser.add_argument('--val_dir_patch', default='./data/validation/patches_256_128')
    # parser.add_argument('--val_dir', default='./data/validation/')
    # parser.add_argument('--val_cam_dir', default='./pmask_cam_ds/validation')

    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()
    args.val_pmask_dir = os.path.join(args.pmask_dir, 'validation')

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.CRITICAL, 
        filename=f"{str(args.save_dir)}/revise_train.log", 
        filemode='w',
        format="%(asctime)s - %(name)s - %(levelname)-9s - %(filename)-8s : %(lineno)s line - %(message)s - %(funcName)s",
        datefmt="%Y-%m-%d %H:%M:%S" 
    )

    logging.critical(args)

    main(args)


