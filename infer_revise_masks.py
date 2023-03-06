import argparse
import os

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8' # for reproducibility
import numpy as np
import random

import torch
import torchvision.transforms

from torch import nn
import torch.nn.functional as F

from pathlib import Path
from PIL import Image

import utils

from tqdm import tqdm
from models.revise_net import Net

import time
execution_time = time.strftime("%Y%m%d%H%M%S", time.localtime())

torch.set_printoptions(precision=8)


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
        
        image = np.array(Image.open(train_image_path).resize((256, 256))) # [256, 256, 3]
        image = self.transforms(image)
        
        pmask = torch.load(os.path.join(self.pmask_dir, name + '.pt'), map_location='cpu') # [32, 32]
        
        cam = np.load(os.path.join(self.cam_dir, name + '.npy')) # [32, 32]
        cam = torch.from_numpy(cam).to(torch.float32)

        if args.dataset == 'wsss4luad':
            label = np.array(utils.to_list(utils.get_label(str(train_image_path))))
        else:
            filename = str(train_image_path)
            label_str = filename.split(']')[0].split('[')[-1]
            l = [int(label_str[0]),int(label_str[1]),int(label_str[2]),int(label_str[3])]
            label = np.array(l)

        
        assert not torch.any(torch.isnan(image)), f'image: {image}'
        assert not torch.any(torch.isnan(pmask)), f'pmask: {pmask}'
        assert not torch.any(torch.isnan(cam)), f'cam: {cam}'
        
        return {'image': image, 'pmask': pmask, 'cam': cam, 'label': torch.Tensor(label), 'name': name}

    def __len__(self):
        return len(self.train_images)
    
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
    dataloader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        worker_init_fn=seed_worker, 
        generator=g, 
        num_workers=args.num_workers
    )
    
    model = Net(num_classes=args.n_class+1).cuda()
    
    model = torch.nn.DataParallel(model).cuda()
    model.load_state_dict(torch.load(args.checkpoint))
    
    infer(dataloader, model, args)

@torch.no_grad()
def infer(dataloader, model, args):
    model.eval()
    with tqdm(dataloader, total=len(dataloader)) as pbar:
        for idx, batch in enumerate(pbar):
            x = batch['image'].cuda()
            N, C, H, W = x.size()
            name_batch = batch['name']

            pmask = batch['pmask']
            pcam = batch['cam']
            n, c, h, w = pmask.size()
            pmask = torch.concat([torch.zeros((n, 1, h, w)), pmask], dim=1).cuda()
            pcam = torch.concat([torch.zeros((n, 1, h, w)), pcam], dim=1).cuda()

            label = batch['label']
            bg_score = torch.ones((n, 1))
            label = torch.cat((bg_score, label), dim=1)
            label = label.cuda(non_blocking=True).unsqueeze(2).unsqueeze(3) # [B, 4, 1, 1]

            _, cam_rv, pmask_rv, pcam_rv = model(x, pmask, pcam)

            pmask_rv = (pmask_rv * label)[:, 1:, :, :]
            pcam_rv = (pcam_rv * label)[:, 1:, :, :]
            cam_rv = (cam_rv * label)[:, 1:, :, :]

            pmask_rv_masks = torch.argmax(pmask_rv, dim=1).cpu().numpy()
            pcam_rv_masks = torch.argmax(pcam_rv, dim=1).cpu().numpy()
            cam_rv_masks = torch.argmax(cam_rv, dim=1).cpu().numpy()

            for pmask_rv_mask, pcam_rv_mask, cam_rv_mask, name in zip(pmask_rv_masks, pcam_rv_masks, cam_rv_masks, name_batch):
                original_image = np.array(Image.open(os.path.join(args.train_dir, name+'.png')))
                h, w, _ = original_image.shape

                if args.dataset == 'wsss4luad':
                    palette = [0, 64, 128, 64, 128, 0, 243, 152, 0, 255, 255, 255] + [0] * 252 * 3
                    background = utils.get_background(original_image)


                    pmask_rv_mask = np.array(Image.fromarray(np.uint8(pmask_rv_mask), mode='P').resize((w, h), resample=Image.BILINEAR))
                    pmask_rv_mask[background > 0] = 3
                    pmask_rv_mask = Image.fromarray(np.uint8(pmask_rv_mask), mode='P')
                    pmask_rv_mask.putpalette(palette)
                    
                    save_dir = '/'.join(args.checkpoint.split('/')[:-1]) # os.path.join(args.pmask_dir, f'25:{args.cam_dir}_val')

                    if not os.path.exists(os.path.join(save_dir, 'refine', 'pmask')):
                        os.makedirs(os.path.join(save_dir, 'refine', 'pmask'))
                    pmask_rv_mask.save(os.path.join(save_dir, 'refine', 'pmask', name+'.png'))

                    pcam_rv_mask = np.array(Image.fromarray(np.uint8(pcam_rv_mask), mode='P').resize((w, h), resample=Image.BILINEAR))
                    pcam_rv_mask[background > 0] = 3
                    pcam_rv_mask = Image.fromarray(np.uint8(pcam_rv_mask), mode='P')
                    pcam_rv_mask.putpalette(palette)
                    if not os.path.exists(os.path.join(save_dir, 'refine', 'pcam')):
                        os.makedirs(os.path.join(save_dir, 'refine', 'pcam'))
                    pcam_rv_mask.save(os.path.join(save_dir, 'refine', 'pcam', name+'.png'))

                    
                    cam_rv_mask = np.array(Image.fromarray(np.uint8(cam_rv_mask), mode='P').resize((w, h), resample=Image.BILINEAR))
                    cam_rv_mask[background > 0] = 3
                    cam_rv_mask = Image.fromarray(np.uint8(cam_rv_mask), mode='P')
                    cam_rv_mask.putpalette(palette)
                    if not os.path.exists(os.path.join(save_dir, 'refine', 'cam')):
                        os.makedirs(os.path.join(save_dir, 'refine', 'cam'))
                    cam_rv_mask.save(os.path.join(save_dir, 'refine', 'cam', name+'.png'))

                else:

                    palette = [0]*15
                    palette[0:3] = [255, 0, 0]
                    palette[3:6] = [0,255,0]
                    palette[6:9] = [0,0,255]
                    palette[9:12] = [153, 0, 255]
                    palette[12:15] = [255, 255, 255]

                    pmask_rv_mask = Image.fromarray(np.uint8(pmask_rv_mask), mode='P').resize((w, h), resample=Image.BILINEAR)
                    pmask_rv_mask.putpalette(palette)
                    
                    save_dir = '/'.join(args.checkpoint.split('/')[:-1]) # os.path.join(args.pmask_dir, f'25:{args.cam_dir}_val')

                    if not os.path.exists(os.path.join(save_dir, 'refine', 'pmask')):
                        os.makedirs(os.path.join(save_dir, 'refine', 'pmask'))
                    pmask_rv_mask.save(os.path.join(save_dir, 'refine', 'pmask', name+'.png'))

                    pcam_rv_mask = Image.fromarray(np.uint8(pcam_rv_mask), mode='P').resize((w, h), resample=Image.BILINEAR)
                    pcam_rv_mask.putpalette(palette)
                    if not os.path.exists(os.path.join(save_dir, 'refine', 'pcam')):
                        os.makedirs(os.path.join(save_dir, 'refine', 'pcam'))
                    pcam_rv_mask.save(os.path.join(save_dir, 'refine', 'pcam', name+'.png'))

                    cam_rv_mask = Image.fromarray(np.uint8(cam_rv_mask), mode='P').resize((w, h), resample=Image.BILINEAR)
                    cam_rv_mask.putpalette(palette)
                    if not os.path.exists(os.path.join(save_dir, 'refine', 'cam')):
                        os.makedirs(os.path.join(save_dir, 'refine', 'cam'))
                    cam_rv_mask.save(os.path.join(save_dir, 'refine', 'cam', name+'.png'))


def parse_args():

    # CUDA_VISIBLE_DEVICES=6, python infer_revise_masks.py --checkpoint weights/miou=0.5782_20220726211929/miou=0.6898601925328832_ResNet38-RFM.pth --pmask_dir /data115_1/fzj/data/pmasks/miou=0.5782_20220726211929 --cam_dir ./pmask_cam_ds/cam

    parser = argparse.ArgumentParser(description='Infer Pseudo-Labels for dataset')

    parser.add_argument('--dataset', type=str, default='wsss4luad')
    parser.add_argument('--n_class', type=int, default=3)

    parser.add_argument('--checkpoint', '-ckpt', type=str)
    parser.add_argument('--batch_size', type=int, default=8)
    
    parser.add_argument('--train_dir', default='./data/training')
    parser.add_argument('--pmask_dir')
    parser.add_argument('--cam_dir', default='./pmask_cam_ds/cam')

    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()
    args.val_patch_dir = os.path.join(args.pmask_dir, 'validation')
    main(args)


