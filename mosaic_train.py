"""
Stage 1 - Train on dataset with Mosaic
"""
import argparse
from functools import partial
import logging
import copy
from pathlib import Path
import random
import shutil
import inspect
import ttach as tta

from models.mosaic_module import MosaicModule

import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import ConcatDataset


import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import loggers as pl_loggers

from dataset import *
import os

import time
execution_time = time.strftime("%Y%m%d%H%M%S", time.localtime())

cpu_num = '2'
os.environ['OMP_NUM_THREADS'] = cpu_num
os.environ['OPENBLAS_NUM_THREADS'] = cpu_num
os.environ['MKL_NUM_THREADS'] = cpu_num
os.environ['VECLIB_MAXIMUM_THREADS'] = cpu_num
os.environ['NUMEXPR_NUM_THREADS'] = cpu_num
torch.set_num_threads(int(cpu_num))

# os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# torch.autograd.set_detect_anomaly(True)

def parse_args():
    # python mosaic_train.py --model=UnetPlusPlus --encoder=efficientnet-b6 --lr=0.001 --gpus=1, --epochs=15 --batch-size=16 --mosaic-data=data/BCSS-WSSS/mosaic_2_112 --patch-size=224 --val-data data/BCSS-WSSS/val --num-classes 4 --dataset bcss

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='the model for training')
    parser.add_argument('--encoder', help='the encoder of the model', default='efficientnet-b6')
    parser.add_argument('--num-classes', help='categories in prediction mask', type=int, default=3)
    parser.add_argument('--dataset', type=str, default='wsss4luad')

    parser.add_argument('--log_dir', default='mosaic_logs')

    parser.add_argument('--tta', action='store_true', default=False)


    parser.add_argument('--patch-size', type=int, default=256)
    
    parser.add_argument('--mosaic-data', default='./data/mosaic')
    parser.add_argument('--val-data', default='./data/validation')
    parser.add_argument('--test-data', default='./data/testing')

    parser.add_argument('--gpus', default=[1,])
    parser.add_argument('--epochs', type=int, default=100)

    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--pin-memory', action='store_true', default=False)

    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')

    args = parser.parse_args()

    return args

#---->load Loggers
def load_loggers(args, name):
    args.log_path = Path(args.log_dir)
    (args.log_path / 'code').mkdir(parents=True, exist_ok=True)

    shutil.copyfile(__file__, args.log_path / 'code' / 'mosaic_train.py')
    shutil.copyfile(inspect.getfile(MosaicModule), args.log_path / 'code' / 'mosaic_module.py')
    shutil.copyfile(inspect.getfile(MosaicDataset), args.log_path / 'code' / 'dataset.py')
    shutil.copyfile('run.sh', args.log_path / 'code' / 'run.sh')

    logging.basicConfig(
        level=logging.CRITICAL, 
        filename=f"{str(args.log_path)}/mosaic_train.log", 
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

    # ----> Build Train Dataset
    train_dataset = []
    train_dataset.append(MosaicDataset(args))

    train_dataset = ConcatDataset(train_dataset)
    
    g = torch.Generator()
    g.manual_seed(0)
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, 
        shuffle=True, num_workers=args.num_workers,
        pin_memory=args.pin_memory, drop_last=True,
        # worker_init_fn=seed_worker, generator=g,
    )

    # ----> Build Validation Dataset
    val_dataset = ValidationDataset(args, args.val_data)

    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers, 
        pin_memory=args.pin_memory
    )    

    # ----> loggers and callbacks
    tb_logger = load_loggers(args, experiment_name)
    callbacks = load_callbacks(args)

    # ----> Model
    model = MosaicModule(args)

    # ----> Trainer
    if 'Unet' in args.model:
        trainer = pl.Trainer(
            logger=tb_logger, 
            max_epochs=args.epochs, 
            gpus=args.gpus, 
            callbacks=callbacks,
            deterministic=True,
        )
    else:
        trainer = pl.Trainer(
            logger=tb_logger, 
            max_epochs=args.epochs, 
            gpus=args.gpus, 
            callbacks=callbacks,
            # deterministic=True,
        )
    
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    
    # ----> Testing with the best model in mIoU
    logging.critical(f'best path: {callbacks[0].best_model_path}')
    args.tta = True # set to True to test with TTA
    model = MosaicModule.load_from_checkpoint(callbacks[0].best_model_path, args=args)

    test_dataset = ValidationDataset(args, args.val_data)
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=64, 
        num_workers=args.num_workers, 
        pin_memory=args.pin_memory)
    test_result = trainer.validate(model, test_dataloader)
    logging.critical(f'Stage 1 - Validation Result: {test_result}')


def get_experiment_name(args):
    experiment_name = f'{args.model}:{args.encoder}:{args.patch_size}:{args.batch_size}:{args.lr}'
    return experiment_name


if __name__ == '__main__':
    args = parse_args()
    # args.execution_time = execution_time
    pl.seed_everything(42)

    experiment_name = get_experiment_name(args)

    main(args, experiment_name)
