import argparse
import random
import os
import re
import time
from pathlib import Path

import torch
from torch.utils.data.dataloader import DataLoader
import torch.nn.functional as F

import pytorch_lightning as pl

from tqdm import tqdm

from dataset import *
from models.mosaic_module import MosaicModule
import ttach as tta

execution_time = time.strftime("%Y%m%d%H%M%S", time.localtime())

cpu_num = '2'
os.environ['OMP_NUM_THREADS'] = cpu_num
os.environ['OPENBLAS_NUM_THREADS'] = cpu_num
os.environ['MKL_NUM_THREADS'] = cpu_num
os.environ['VECLIB_MAXIMUM_THREADS'] = cpu_num
os.environ['NUMEXPR_NUM_THREADS'] = cpu_num
torch.set_num_threads(int(cpu_num))

# os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'

def parse_args():

    # CUDA_VISIBLE_DEVICES=4, python infer_pseudo_masks.py --checkpoint bcss_logs/Unet:efficientnet-b0:224:16:0.001 --train-data data115/BCSS-WSSS/training --save-dir bcss_logs --gpus 0 --batch-size 64 --dataset bcss

    parser = argparse.ArgumentParser(description='Infer Pseudo-Labels for dataset')

    parser.add_argument('--checkpoint', '-ckpt', help='the checkpoint path of stage 1 model')

    parser.add_argument('--train-data', default='./data/training')
    parser.add_argument('--save-dir', default='./pmask')
    parser.add_argument('--gpus', type=int)

    parser.add_argument('--dataset', type=str, default='wsss4luad')

    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--pin-memory', action='store_true', default=False)

    parser.add_argument('--patch-size', type=int, default=256)


    args = parser.parse_args()

    return args

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32    
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def check_tissue_region_is_too_small(patch_mask_pred, patch_label):
    for i in range(len(patch_label)):
        if patch_label[i] == 1:
            if np.sum(patch_mask_pred == i) / (patch_mask_pred.shape[-2] * patch_mask_pred.shape[-1]) < 0.1:
                return True
    return False

def get_mask_pred_and_entropy(patch_logit_pred, tissue, patch_label):
    # there is only one tissue type in the patch
    if sum(patch_label) == 1:
        mask_pred = np.full((patch_logit_pred.shape[-2], patch_logit_pred.shape[-1]), patch_label.index(1))
        entropy = np.zeros_like(mask_pred)
    else:
        # there are multiple tissue types in the patch
        for i in range(len(patch_label)):
            if patch_label[i] == 0: # do not have this tissue
                patch_logit_pred[i,:,:] = -1e10 # [C, H, W] # Warning: inplace operation!
        
        patch_pos_pred = torch.softmax(patch_logit_pred, dim=0)
        entropy = -torch.sum(patch_pos_pred * torch.log(patch_pos_pred + 1e-10), dim=0).cpu().numpy() # [H, W]
        mask_pred = torch.argmax(patch_pos_pred, dim=0)
        mask_pred = mask_pred.cpu().numpy()
    
    mask_pred[tissue == 0] = len(patch_label)

    return mask_pred, entropy

def interpolate_tensor(tensor, target_shape):
    return F.interpolate(tensor.unsqueeze(0), target_shape, mode='bilinear')[0]

def main(args):

    # Stage 2 - Inference Pseudo-Labels for dataset
    model = MosaicModule.load_from_checkpoint(args.checkpoint).cuda(args.gpus)
    model = tta.SegmentationTTAWrapper(model, tta.aliases.d4_transform(), merge_mode='mean')
    
    train_dataset = TrainDataset(args)
    g = torch.Generator()
    g.manual_seed(0)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=args.pin_memory, shuffle=False, worker_init_fn=seed_worker, generator=g,)


    if not os.path.exists(os.path.join(args.save_dir, 'mask')):
        (Path(args.save_dir) / 'mask').mkdir(parents=True, exist_ok=True)

    if not os.path.exists(os.path.join(args.save_dir, 'logits_32x32')):
        (Path(args.save_dir) / 'logits_32x32').mkdir(parents=True, exist_ok=True)

    if not os.path.exists(os.path.join(args.save_dir, 'background-img')):
        (Path(args.save_dir) / 'background-img').mkdir(parents=True, exist_ok=True)
    
    if not os.path.exists(os.path.join(args.save_dir, 'entropy')):  
        (Path(args.save_dir) / 'entropy').mkdir(parents=True, exist_ok=True)

    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(train_dataloader)):
            image_batch, tissue_batch, name_batch = batch['image'].cuda(args.gpus), batch['tissue'], batch['name']

            logit_pred = model(image_batch)

            for patch_logit_pred, tissue, name in zip(logit_pred, tissue_batch, name_batch):

                # ----> Save reshaped logits (32 x 32)
                patch_logit_pred_reshaped = interpolate_tensor(patch_logit_pred, (32, 32))
                torch.save(patch_logit_pred_reshaped, Path(args.save_dir) / 'logits_32x32' / (name.split('.png')[0] + '.pt'))

                # ----> Get Auxiliary Information
                if args.dataset == 'wsss4luad':
                    patch_label = utils.to_list(utils.get_label(name))
                else:
                    label_str = name.split(']')[0].split('[')[-1]
                    patch_label = [int(label_str[0]),int(label_str[1]),int(label_str[2]),int(label_str[3])]

                tissue = tissue.numpy()
                patch_mask_pred, entropy = get_mask_pred_and_entropy(patch_logit_pred, tissue, patch_label)

                w, h = Image.open(Path(args.train_data) / name).size

                # ----> Save reshaped predicted mask (original size)
                if args.dataset == 'wsss4luad':
                    palette = [0, 64, 128, 64, 128, 0, 243, 152, 0, 255, 255, 255] + [0] * 252 * 3
                else:
                    palette = [0]*15
                    palette[0:3] = [255, 0, 0]
                    palette[3:6] = [0,255,0]
                    palette[6:9] = [0,0,255]
                    palette[9:12] = [153, 0, 255]
                    palette[12:15] = [255, 255, 255]
                patch_mask_pred = Image.fromarray(np.uint8(patch_mask_pred), mode='P')
                patch_mask_pred.putpalette(palette)
                patch_mask_pred = patch_mask_pred.resize((w, h), resample=Image.BILINEAR)
                patch_mask_pred.save(Path(args.save_dir) / 'mask' / name)

                # ----> Save reshaped entropy (original size) and convert to [0, 255] (legacy, no use)
                # entropy = Image.fromarray(np.uint8(255 * entropy), mode='L')
                # entropy = entropy.resize((w, h), resample=Image.BILINEAR)
                # entropy.save(Path(args.save_dir) / 'entropy' / name)


if __name__ == '__main__':
    args = parse_args()
    print('Saving to {}'.format(args.save_dir))

    for filename in os.listdir(args.checkpoint):
        if 'epoch=' in filename:
            checkpoint_file_path = os.path.join(args.checkpoint, filename)
            break
    else:
        assert False, 'Cannot find a valid checkpoint file in {args.checkpoint}'

    args.checkpoint = checkpoint_file_path
    print(f'Loading checkpoint from {args.checkpoint}')

    pl.seed_everything(42, workers=True)
    main(args)
