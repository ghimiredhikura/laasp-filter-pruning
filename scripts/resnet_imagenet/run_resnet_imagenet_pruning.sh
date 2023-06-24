
IMAGENET_PATH="C:/ImageNet"
PRUNE_DIR="D:/Soongsil/imagenet_pruning/movilenet"

#train_prune_baseline_resnet_imagenet prune resnet18 $PRUNE_DIR/$PRUNE_EPOCH.resnet18_0.420 0.420
#train_prune_baseline_resnet_imagenet prune resnet34 $PRUNE_DIR/$PRUNE_EPOCH.resnet34_0.420 0.420
#train_prune_baseline_resnet_imagenet prune resnet50 $PRUNE_DIR/$PRUNE_EPOCH.resnet50_0.420 0.420


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

#PRUNE_EPOCH="25"
#train_prune_resnet_imagenet prune mobilenet_v2 $PRUNE_DIR/$PRUNE_EPOCH.mobilenet_v2_0.55 0.55

#OMP_NUM_THREADS=8 torchrun --nproc_per_node=2 --nnodes=1 --master_port 18115 \
train_model_imagenet()
{    
    python main_resnet_imagenet.py --data_path $IMAGENET_PATH \
    --mode $1 \
    --arch $2 \
    --save_path $3 \
    --start_epoch 58 \
    --total_epoches 100 \
    --decay_epoch_step 30 \
    --workers 4 \
    --lr 0.1 --decay 0.0001 --batch_size 256 \
    --pretrain_path ""D:/Soongsil/imagenet_pruning/movilenet/.mobilenet_v2_0.55/mobilenet_v2.checkpoint.pth.tar""
}

train_model_imagenet train mobilenet_v2 $PRUNE_DIR/$PRUNE_EPOCH.mobilenet_v2_0.55