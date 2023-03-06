# Genral Config
GPU=7,

# ----> Dataset Paths
MOSAIC_DATA=mosaic_7_32
TRAIN_DATA_PATH=data/WSSS4LUAD/1.training
MOSAIC_DATA_PATH=data/WSSS4LUAD/$MOSAIC_DATA
VAL_PATCH_PATH=data/WSSS4LUAD/2.validation/patches_224_56
TEST_PATCH_PATH=data/WSSS4LUAD/3.testing/patches_224_112
VAL_PATH=data/WSSS4LUAD/2.validation

# ----> Mosaic Training and Revise Arguments
MODEL=UnetPlusPlus
ENCODER=efficientnet-b0
EPOCHS=15 # set to 15
PATCH_SIZE=224
BATCH_SIZE=16
LR=0.001

# ----> Revise Module Arguments
REVISE_EPOCHS=25 # set to 25
CAM_PATH=data/WSSS4LUAD/CAM/train # data115/WSSS4LUAD/CAM/cam # data115/sc-cam/train

# log name
MOSAIC_LOG_DIR=./logs

EXPERIMEN_NAME=$MOSAIC_DATA:$MODEL:$ENCODER:$PATCH_SIZE:$BATCH_SIZE:$LR
MOSAIC_CKPT_PATH=$MOSAIC_LOG_DIR/$EXPERIMEN_NAME

# ----> Segmentation Arguments
REVISE_MASK_TYPE=cam #cam, pcam, pmask
SEGMENTATION_MODEL=UnetPlusPlus
SEGMENTATION_ENCODER=efficientnet-b3
SEGMENTATION_LR=0.0005
SEGMENTATION_EPOCHS=15 # set to 15

SEGMENTATION_PATCH_SIZE=$PATCH_SIZE

SEGMENTATION_EXPERIMENT_NAME=$SEGMENTATION_MODEL:$SEGMENTATION_ENCODER:$SEGMENTATION_PATCH_SIZE:$BATCH_SIZE:$SEGMENTATION_LR
# =====> End of arguments <====

# Train with Mosaic Dataset
echo "Train with Mosaic Dataset"
python mosaic_train.py --model=$MODEL --encoder=$ENCODER --lr=$LR --gpus=$GPU --epochs=$EPOCHS --batch-size=$BATCH_SIZE --mosaic-data=$MOSAIC_DATA_PATH --patch-size=$PATCH_SIZE --val-data=data/WSSS4LUAD/2.validation/patches_224_112 --log_dir $MOSAIC_CKPT_PATH

# Generate Training Logits and masks
echo "Generate Training Logits and masks"
CUDA_VISIBLE_DEVICES=$GPU python infer_pseudo_masks.py --checkpoint $MOSAIC_CKPT_PATH --train-data $TRAIN_DATA_PATH --save-dir $MOSAIC_CKPT_PATH --gpus 0 --batch-size $BATCH_SIZE

# Train Mask Revise Module
echo "Train Mask Revise Module"
CUDA_VISIBLE_DEVICES=$GPU python revise_pseudo_labels.py --pmask_dir $MOSAIC_CKPT_PATH --cam_dir $CAM_PATH --train_dir $TRAIN_DATA_PATH --max_epoches $REVISE_EPOCHS --lr 1e-3 --save_dir $MOSAIC_CKPT_PATH

# Infer revised masks
echo "Infer revised masks"
CUDA_VISIBLE_DEVICES=$GPU python infer_revise_masks.py --train_dir $TRAIN_DATA_PATH --checkpoint $MOSAIC_CKPT_PATH/ResNet38-RFM.pth --pmask_dir $MOSAIC_CKPT_PATH --cam_dir $CAM_PATH --batch_size 64

# Train Segmentation Model
echo "Train Segmentation Model"
python segmentation_train.py --model $SEGMENTATION_MODEL --encoder $SEGMENTATION_ENCODER --lr $SEGMENTATION_LR --gpus=$GPU --epochs=$SEGMENTATION_EPOCHS --batch-size=$BATCH_SIZE --pseudo-mask-dir=$MOSAIC_CKPT_PATH/refine/$REVISE_MASK_TYPE --patch-size=$SEGMENTATION_PATCH_SIZE --train-data $TRAIN_DATA_PATH --val-data=$VAL_PATCH_PATH  --log_dir $MOSAIC_CKPT_PATH/segmentation_log/$SEGMENTATION_EXPERIMENT_NAME

# Test Segmentation Model
echo "Test Segmentation Model"
CUDA_VISIBLE_DEVICES=$GPU python segmentation_test.py -ckpt $MOSAIC_CKPT_PATH/segmentation_log/$SEGMENTATION_EXPERIMENT_NAME --gpus=0, --patch-size=$SEGMENTATION_PATCH_SIZE --test-data $TEST_PATCH_PATH

# ======================== Note ========================
# Test masks will be saved in $MOSAIC_CKPT_PATH/segmentation_log/$SEGMENTATION_EXPERIMENT_NAME/test

# Test IoU will be displayed in the terminal
# mIoU(big mask): xxxxxxx
# fwIoU: xxxxx
# tIoU, sIoU, nIoU: [xxxx, xxxx, xxxx]
