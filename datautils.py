import math
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
from tqdm import tqdm as tqdm

def val_collate_fn(args):
    def collate_fn(batch):
        h_max = w_max = 0
        for image, _, _ in batch:
            h, w = image.shape[-2:]
            h_max = max(h_max, h)
            w_max = max(w_max, w)
        if h_max <= args.patch_size and w_max <= args.patch_size:
            height = args.patch_size
            width = args.patch_size
            # height = h_max if h_max % 32 == 0 else (h_max // 32 + 1) * 32
            # width = w_max if w_max % 32 == 0 else (w_max // 32 + 1) * 32
        else:
            height = (args.patch_size - args.patch_step) + math.ceil((h_max - (args.patch_size - args.patch_step)) / args.patch_step) * args.patch_step
            width = (args.patch_size - args.patch_step) + math.ceil((w_max - (args.patch_size - args.patch_step)) / args.patch_step) * args.patch_step

        images, masks, names = [], [], []
        if args.collate_fn == 'padding':
            for image, mask, name in batch:
                h, w = image.shape[-2:]
                if args.padding_mode == 'constant':
                    image_padded = T.Pad([0, 0, width-w, height-h], padding_mode=args.padding_mode)(image)
                    mask_padded = T.Pad([0, 0, width-w, height-h], padding_mode=args.padding_mode, fill=3)(mask)
                else:
                    image_padded = T.Pad([0, 0, width-w, height-h], padding_mode=args.padding_mode)(image)
                    mask_padded = T.Pad([0, 0, width-w, height-h], padding_mode=args.padding_mode)(mask)
                images.append(image_padded)
                masks.append(mask_padded)
                names.append(name)
        elif args.collate_fn == 'resize':
            for image, mask, name in batch:
                image_resized = T.Resize((height, width), interpolation=T.InterpolationMode.BICUBIC)(image)
                # INFO TODO: a hack
                mask_padded = T.Pad([0, 0, width-w, height-h], fill=3)(mask)
                images.append(image_resized)
                masks.append(mask_padded)
                names.append(name)

        images = torch.stack(images)
        masks = torch.stack(masks)
        return images, masks, names
    return collate_fn

def test_padding(batch):
    # batch: [N, 3, C, H, W]
    images, sizes, names, imgs_not_normalized, backgrounds = [], [], [], [], []
    max_height, max_width = 0, 0

    # iter the batch dim
    for sample in batch:
        image, size, name, img_not_normalized, background = sample[0], sample[1], sample[2], sample[3], sample[4]

        max_height = max(max_height, image.shape[1])
        max_width = max(max_width, image.shape[2])

        images.append(image)
        sizes.append(size)
        names.append(name)
        imgs_not_normalized.append(img_not_normalized)
        backgrounds.append(background)
        
    if max_height <= 256 and max_width <= 256:
        height = max_height if max_height % 32 == 0 else (max_height // 32 + 1) * 32
        width = max_width if max_width % 32 == 0 else (max_width // 32 + 1) * 32
    else:
        height = max_height if max_height % 256 == 0 else (max_height // 256 + 1) * 256
        width = max_width if max_width % 256 == 0 else (max_width // 256 + 1) * 256

    pad_imgs = []
    pad_imgs_not_normalized = []
    pad_bkgs = []

    for img, img_not_normalized, bkg in zip(images, imgs_not_normalized, backgrounds):

        img = img.unsqueeze(0) # [1, C, H, W]
        img_not_normalized = img_not_normalized.unsqueeze(0) # [1, C, H, W]
        bkg = bkg.unsqueeze(0).unsqueeze(0) # [1, 1, H, W]

        img_padded = F.pad(img, (0, width - img.shape[-1], 0, height - img.shape[-2]), mode='replicate')
        
        pad_img_not_normalized = F.pad(img_not_normalized, (0, width - img_not_normalized.shape[-1], 0, height - img_not_normalized.shape[-2]), mode='replicate')
        
        bkg_padded = F.pad(bkg, (0, width - bkg.shape[-1], 0, height - bkg.shape[-2]), mode='replicate')


        pad_imgs.append(img_padded.squeeze())
        pad_imgs_not_normalized.append(pad_img_not_normalized.squeeze())
        pad_bkgs.append(bkg_padded.squeeze())

    images = torch.stack(pad_imgs)
    sizes = np.stack(sizes)
    names = np.stack(names)
    imgs_not_normalized = torch.stack(pad_imgs_not_normalized)
    backgrounds = torch.stack(pad_bkgs)

    return images, sizes, names, imgs_not_normalized, backgrounds