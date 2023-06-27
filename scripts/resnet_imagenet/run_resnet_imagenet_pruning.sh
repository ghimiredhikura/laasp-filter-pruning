
IMAGENET_PATH="C:/ImageNet"
PRUNE_DIR="imagenet_pruning"

train_prune_resnet_imagenet()
{
    python main_resnet_imagenet.py --data_path $IMAGENET_PATH \
    --mode $1 \
    --arch $2 \
    --save_path $3 \
    --rate_flop $4 \
    --start_epoch 0 \
    --total_epoches 100 \
    --recover_epoch 2 \
    --prune_epoch $PRUNE_EPOCH \
    --decay_epoch_step 30 \
    --workers 8 \
    --lr 0.1 --decay 0.0001 --batch_size 256
}

PRUNE_EPOCH="25"
train_prune_resnet_imagenet prune resnet50 $PRUNE_DIR/$PRUNE_EPOCH.resnet50_0.58 0.58