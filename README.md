# PistoSeg
Code Repository for AAAI23 Paper "Weakly-Supervised Semantic Segmentation for Histopathology Images Based on Dataset Synthesis and Feature Consistency Constraint"

## Installation
Code tested on
* Ubuntu 18.04
* A single Nvidia GeForce RTX 3090
* Python 3.8
* Pytorch 1.12.1
* Pytorch Lightning 1.7.1

Please use the follwing command to install the dependencies:

`conda env create -f environment.yaml`

## Prepraing the Data and Weights

1. Download the [WSSS4LUAD dataset](https://wsss4luad.grand-challenge.org/) and put it in ./data/WSSS4LUAD

2. Download the [BCSS-WSSS dataset](https://drive.google.com/drive/folders/1iS2Z0DsbACqGp7m6VDJbAcgzeXNEFr77) and put it in ./data/BCSS-WSSS (Thanks to [Han et. al](https://github.com/ChuHan89/WSSS-Tissue))

2. Download the ImageNet-pretrained ResNet weight `ilsvrc-cls_rna-a1_cls1000_ep-0001.params` from [SEAM responsitory](https://github.com/YudeWang/SEAM) and put it in ./weights/ilsvrc-cls_rna-a1_cls1000_ep-0001.params

3. Download the CAM model's weights `res38d.pth` from [OEEM responsitory](https://github.com/xmed-lab/OEEM) and put it in ./weights/res38d.pth

## CAM Generation
Plese refer to the OEEM readme in ./OEEM/README.md

Or you can use the [pre-generated CAMs](https://pan.baidu.com/s/1lcC1c04gZjiujR2xrdfUng?pwd=yj84) (in AAAI23/{dataset}/data/CAM/train.zip) and unzip and put them in ./data/WSSS4LUAD/CAM/train and ./data/BCSS-WSSS/CAM/train.

## Train & Test

### WSSS4LUAD

1. Assure that the CAM of the training set is put into ./data/WSSS4LUAD/CAM/train (in .npy format)

2. Split the validation and test dataset with `split_validation.ipynb` to produce 224x224 regular-shaped patches

3. Prepare the synthesized dataset and background mask with `create_dataset.ipynb`

4. Run `bash run.sh`


### BCSS-WSSS

1. Assure that the CAM of the training set is put into ./data/BCSS-WSSS/CAM/train (in .npy format)

2. Prepare the synthesized dataset with `create_dataset_bcss.ipynb`

3. run `bash run-bcss.sh`

## Note
Due to a disaster to the server, the original weights of the results provided in the paper are lost. So the current codes and weights are re-implemented and re-trained. The results are a bit different from the original ones (but slightly improved in mIoU). The overall performance is still similar.

### WSSS4LUAD
* Test mIoU: 0.7530
* Test fwIoU: 0.7582
* Test tissue IoU: [0.7991 0.7020 0.7580] - [TUM, STR, NOM]

### BCSS-WSSS
* Test mIoU: 0.7075
* Test fwIoU: 0.7576
* Test tissue IoU: [0.8144 0.7446 0.6063 0.6645] - [TUM, STR, LYM, NEC]

### Reproducibility
We tried our best to ensure the reproducibility of the results, but since the `torch.nn.functional.interpolate` function is **not deterministic**, the results may be different over runs. If you want to fully reproduce the results, you can use the [following weights](https://pan.baidu.com/s/1lcC1c04gZjiujR2xrdfUng?pwd=yj84) (**code: yj84**) (preliminary segmentation `epoch=*.ckpt`, refining `ResNet38-RFM.pth`, and precise segmentation `segmentation_log/epoch=*.ckpt`) and intermediate results (Generated CAM `data/CAM/train.zip` and Refined Masks `refine/CAM.zip`). Training logs are also provided for reference.

 




