import os
from PIL import Image
import numpy as np
from multiprocessing import Array, Process
from utils.pyutils import chunks
import random

def calculate_F1(pred_path, gt_path, numofclass):
    TPs = [0] * numofclass
    FPs = [0] * numofclass
    FNs = [0] * numofclass
    ims = os.listdir(pred_path)
    for im in ims:
        pred = np.asarray(Image.open(os.path.join(pred_path, im)))
        gt = np.asarray(Image.open(os.path.join(gt_path, im)))
        for k in range(numofclass):
            TPs[k] += np.sum(np.logical_and(pred == k, gt == k))
            FPs[k] += np.sum(np.logical_and(pred == k, gt != k))
            FNs[k] += np.sum(np.logical_and(pred != k, gt == k))
 
    f1_score = TPs / (TPs + (FPs + FNs)/2 + 1e-7)
    f1_score = sum(f1_score) / numofclass
    return f1_score


def get_overall_valid_score(pred_image_path, groundtruth_path, num_workers=5, num_class=2):
    """
    get the scores with validation groundtruth, the background will be masked out
    and return the score for all photos

    Args:
        pred_image_path (str): the prediction require to test, npy format
        groundtruth_path (str): groundtruth images, png format
        num_workers (int): number of process in parallel, default is 5.
        mask_path (str): the white background, png format
        num_class (int): default is 2.

    Returns:
        float: the mIOU score
    """
    image_names = list(map(lambda x: '.'.join(x.split('.')[:-1]), os.listdir(pred_image_path)))
    random.shuffle(image_names)
    image_list = chunks(image_names, num_workers)

    def f(intersection, union, image_list):
        gt_list = []
        pred_list = []

        for im_name in image_list:
            cam = np.load(os.path.join(pred_image_path, f"{im_name}.npy"), allow_pickle=True).astype(np.uint8).reshape(-1)
            groundtruth = np.asarray(Image.open(groundtruth_path + f"/{im_name}.png")).reshape(-1)
            
            print(f'cam: {cam.shape}; groundtruth: {groundtruth.shape}')
            print(f'cam path: {os.path.join(pred_image_path, f"{im_name}.npy")}')
            
            gt_list.extend(groundtruth)
            pred_list.extend(cam)

        pred = np.array(pred_list)
        real = np.array(gt_list)
        
        print(f'pred: {pred.shape}; real: {real.shape}')
        
        for i in range(num_class):
            if i in pred:
                inter = sum(np.logical_and(pred == i, real == i))
                u = sum(np.logical_or(pred == i, real == i))
                intersection[i] += inter
                union[i] += u

    intersection = Array("d", [0] * num_class)
    union = Array("d", [0] * num_class)
    p_list = []
    for i in range(len(image_list)):
        p = Process(target=f, args=(intersection, union, image_list[i]))
        p.start()
        p_list.append(p)
    for p in p_list:
        p.join()

    eps = 1e-7
    total = 0
    for i in range(num_class):
        class_i = intersection[i] / (union[i] + eps)
        total += class_i
    return total / num_class
