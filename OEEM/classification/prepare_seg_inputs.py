import argparse
import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from tqdm import tqdm
import torch.nn.functional as F
from dataset import TrainingSetCAM
import network
from utils.pyutils import glas_join_crops_back
import yaml
import importlib

# limit CPU usage
# savely comment the lines below if you want to use all CPU cores
cpu_num = '2'
os.environ['OMP_NUM_THREADS'] = cpu_num
os.environ['OPENBLAS_NUM_THREADS'] = cpu_num
os.environ['MKL_NUM_THREADS'] = cpu_num
os.environ['VECLIB_MAXIMUM_THREADS'] = cpu_num
os.environ['NUMEXPR_NUM_THREADS'] = cpu_num
torch.set_num_threads(int(cpu_num))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-batch', default=20, type=int)
    parser.add_argument('-d','--device', nargs='+', help='GPU id to use parallel', required=True, type=int)
    parser.add_argument('-ckpt', type=str, required=True, help='the checkpoint model name')
    parser.add_argument('-dataset', default='wsss4luad', help='the dataset name')
    
    args = parser.parse_args()

    batch_size = args.batch
    devices = args.device
    ckpt = args.ckpt
    dataset = args.dataset

    with open(f'classification/configuration_{dataset}.yml') as f:
        data = yaml.safe_load(f)
    mean = data['mean']
    std = data['std']
    side_length = data['side_length']
    stride = data['stride']
    num_of_class = data['num_of_class']
    network_image_size = data['network_image_size']
    scales = data['scales']

    train_pseudo_mask_path = f'classification/{dataset}-' + ckpt.replace('.pth', '') + '_train_pseudo_mask' # classification/{dataset}-res38d_train_pseudo_mask
    if not os.path.exists(train_pseudo_mask_path):
        os.mkdir(train_pseudo_mask_path)

    if dataset == 'glas':
        data_path_name = f'classification/glas/1.training/img'
    elif dataset == 'wsss4luad':
        data_path_name = f'classification/WSSS4LUAD/1.training'
    elif dataset == 'bcss':
        data_path_name = f'classification/BCSS-WSSS/training'
    
    dset = TrainingSetCAM(data_path_name=data_path_name, transform=transforms.Compose([
                        transforms.Resize((network_image_size, network_image_size)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=mean, std=std)
                ]), patch_size=side_length, stride=stride, scales=scales, num_class=num_of_class
    )
    dataLoader = DataLoader(dset, batch_size=1, drop_last=False)

    net_cam = getattr(importlib.import_module("network.wide_resnet"), 'wideResNet')(num_class=num_of_class)
    model_path = f"classification/weights/{dataset}/" + ckpt + ".pth"
    pretrained = torch.load(model_path)['model']
    pretrained = {k[7:]: v for k, v in pretrained.items()}
    pretrained['fc_cam.weight'] = pretrained['fc_cls.weight'].unsqueeze(-1).unsqueeze(-1).to(torch.float64)
    pretrained['fc_cam.bias'] = pretrained['fc_cls.bias']
    net_cam.load_state_dict(pretrained)

    net_cam.eval()
    net_cam = torch.nn.DataParallel(net_cam, device_ids=devices).cuda()

    with torch.no_grad():
        for im_name, scaled_im_list, scaled_position_list, scales, big_label in tqdm(dataLoader):
            big_label = big_label[0]
            
            # training images have big labels, can be used to improve CAM performance
            eliminate_noise = True
            if len(big_label) == 1:
                eliminate_noise = False
            
            orig_img = np.asarray(Image.open(f'{data_path_name}/{im_name[0]}'))
            w, h, _ = orig_img.shape


            ensemble_cam = np.zeros((num_of_class, w, h))

            # get the prediction for each pixel in each scale
            for s in range(len(scales)):
                w_ = int(w*scales[s])
                h_ = int(h*scales[s])
                interpolatex = side_length
                interpolatey = side_length

                if w_ < side_length:
                    interpolatex = w_
                if h_ < side_length:
                    interpolatey = h_

                im_list = scaled_im_list[s]
                position_list = scaled_position_list[s]

                im_list = torch.vstack(im_list)
                im_list = torch.split(im_list, batch_size)

                cam_list = []
                for ims in im_list:
                    cam_scores = net_cam.module.forward_cam(ims.cuda())
                    cam_scores = F.interpolate(cam_scores, (interpolatex, interpolatey), mode='bilinear', align_corners=False).detach().cpu().numpy()
                    cam_list.append(cam_scores)
                cam_list = np.concatenate(cam_list)

                sum_cam = np.zeros((num_of_class, w_, h_))
                sum_counter = np.zeros_like(sum_cam)
            
                for k in range(cam_list.shape[0]):
                    y, x = position_list[k][0], position_list[k][1]
                    crop = cam_list[k]
                    sum_cam[:, y:y+side_length, x:x+side_length] += crop
                    sum_counter[:, y:y+side_length, x:x+side_length] += 1
                sum_counter[sum_counter < 1] = 1

                norm_cam = sum_cam / sum_counter
                norm_cam = F.interpolate(torch.unsqueeze(torch.tensor(norm_cam),0), (w, h), mode='bilinear', align_corners=False).detach().cpu().numpy()[0]
                
                # use the image-level label to eliminate impossible pixel classes
                ensemble_cam += norm_cam
            
            ensemble_cam /= len(scales)
            ensemble_cam = F.interpolate(torch.unsqueeze(torch.tensor(ensemble_cam),0), (32, 32), mode='bilinear', align_corners=False).detach().cpu().numpy()[0]
            np.save(f'{train_pseudo_mask_path}/{".".join(im_name[0].split(".")[:-1])}.npy', ensemble_cam)

