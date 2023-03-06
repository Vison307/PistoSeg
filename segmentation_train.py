import argparse
import inspect
import logging
from pathlib import Path
import random
import shutil

from models.segmentation_module import SegmentationModule
import torch
from torch.utils.data.dataloader import DataLoader


import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning import loggers as pl_loggers

from dataset import *
import os


cpu_num = '2'
os.environ['OMP_NUM_THREADS'] = cpu_num
os.environ['OPENBLAS_NUM_THREADS'] = cpu_num
os.environ['MKL_NUM_THREADS'] = cpu_num
os.environ['VECLIB_MAXIMUM_THREADS'] = cpu_num
os.environ['NUMEXPR_NUM_THREADS'] = cpu_num
torch.set_num_threads(int(cpu_num))

# os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'

def parse_args():
    # python segmentation_train.py --model Unet --encoder efficientnet-b3 --lr=5e-4 --gpus=1, --epochs=10 --batch-size=16 --pseudo-mask-dir=/data115_1/fzj/data/pmasks/miou=0.5782_20220726211929/refine/cam --patch-size=256 --patch-step=128 --val-data=/data115_1/fzj/data/testing/patches_256_128_20220730 --train-data=/data115_1/fzj/data/training 

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='the model for training', default="DeepLabV3Plus")
    parser.add_argument('--encoder', help='the encoder of the model', default='efficientnet-b6')
    parser.add_argument('--num-classes', help='categories in prediction mask', type=int, default=3)
    parser.add_argument('--dataset', default='wsss4luad')

    parser.add_argument('--log_dir', default='mosaic_logs')
    
    parser.add_argument('--cutmix-pseudo', action='store_true', default=False)
    parser.add_argument('--cutmix-prob', type=float, default=0.8)

    parser.add_argument('--patch-size', type=int, default=256)

    parser.add_argument('--pseudo-mask-dir')
    parser.add_argument('--train-data', default='./data/training')
    parser.add_argument('--val-data', default='./data/validation')
    parser.add_argument('--test-data', default='./data/testing')

    parser.add_argument('--gpus', default=[1,])
    parser.add_argument('--epochs', type=int, default=100)

    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--pin-memory', action='store_true', default=False)

    # Optimizer parameters
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05, help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',  help='learning rate (default: 5e-4)')

    args = parser.parse_args()

    return args

#---->load Loggers
def load_loggers(args, name):
    
    args.log_path = Path(args.log_dir)
    (args.log_path / 'code').mkdir(parents=True, exist_ok=True)

    shutil.copyfile(__file__, args.log_path / 'code' / 'segmentation_train.py')
    shutil.copyfile(inspect.getfile(SegmentationModule), args.log_path / 'code' / 'segmentation_module.py')
    shutil.copyfile(inspect.getfile(PseudoTrainDataset), args.log_path / 'code' / 'dataset.py')
    shutil.copyfile('./infer_pseudo_masks.py', args.log_path / 'code' / 'infer_pseudo_masks.py')

    logging.basicConfig(
        level=logging.CRITICAL, 
        filename=f"{str(args.log_path)}/segmentation_train.log", 
        filemode='w',
        format="%(asctime)s - %(name)s - %(levelname)-9s - %(filename)-8s : %(lineno)s line - %(message)s - %(funcName)s",
        datefmt="%Y-%m-%d %H:%M:%S" 
    )

    logging.critical(args)


    #---->TensorBoard
    tb_logger = pl_loggers.TensorBoardLogger(
        args.log_dir,
        name='',
        version='',
        default_hp_metric = False)
    #---->CSV
    csv_logger = pl_loggers.CSVLogger(
        args.log_dir,
        name='',
        version='',)
    
    return [tb_logger, csv_logger]

def load_callbacks(args):
    Mycallbacks = []

    Mycallbacks.append(
        ModelCheckpoint(
            monitor='validation_miou_mask_epoch',
            filename='{epoch:02d}-{validation_miou_mask_epoch:.4f}',
            save_last=True,
            verbose = True,
            mode='max',
            dirpath = str(args.log_path),
        )
    )
    Mycallbacks.append(LearningRateMonitor())
            
    return Mycallbacks

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    
def main(args, experiment_name):
    
    tb_logger = load_loggers(args, experiment_name)
    callbacks = load_callbacks(args)

    # Stage 3 - Train a (new) model with the dataset with pseudo labels
    mask_dir = args.pseudo_mask_dir

    # ----> Build Training Dataset
    train_dataset = PseudoTrainDataset(args, mask_dir)
    g = torch.Generator()
    g.manual_seed(0)
    pseudo_train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=args.pin_memory, drop_last=True, worker_init_fn=seed_worker, generator=g) 

    # ----> Build Validation Dataset
    val_dataset = ValidationDataset(args, args.val_data)
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers, 
        pin_memory=args.pin_memory
    )     

    model = SegmentationModule(args)
    model.save_dir = mask_dir

    if 'Unet' in args.model:
        trainer = pl.Trainer(
            logger=tb_logger, 
            max_epochs=args.epochs, 
            gpus=args.gpus, 
            callbacks=callbacks,
            deterministic=True
        )
    else:
        # DeepLabv3+ has upsampling which is not deterministic
        trainer = pl.Trainer(
            logger=tb_logger, 
            max_epochs=args.epochs, 
            gpus=args.gpus, 
            callbacks=callbacks,
            # deterministic=True
        )
    # torch.use_deterministic_algorithms(mode=True, warn_only=True)
    trainer.fit(model, train_dataloaders=pseudo_train_dataloader, val_dataloaders=val_dataloader)



def get_experiment_name(args):
    experiment_name = f'{args.model}:{args.encoder}:{args.patch_size}:{args.batch_size}:{args.lr}'

    return experiment_name


if __name__ == '__main__':
    args = parse_args()
    pl.seed_everything(42, workers=True)

    experiment_name = get_experiment_name(args)
    main(args, experiment_name)
