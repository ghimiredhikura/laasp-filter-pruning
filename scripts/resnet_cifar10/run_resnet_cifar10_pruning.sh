
PRUNE_DIR="resnet_cifar10_prune"

pruning_ratio_pretrain_cifar10_resnet()
{
    python main_resnet_cifar10.py --data_path ./data/cifar.python --dataset cifar10 \
    --mode $1 \
    --arch $2 \
    --save_path $3 \
    --rate_flop $4 \
    --total_epoches 200 \
    --schedule 60 120 160 190 \
    --gammas 0.1 0.1 0.1 0.1 \
    --recover_flop 0.02 \
    --recover_epoch 6 \
    --prune_epoch $5 \
    --workers 6 \
    --max_prune_limit 0.75 \
    --step_scale $6 \
    --lr 0.1 --decay 5e-4 --batch_size 128
}

run_110()
{
    NET="resnet110"
    FLOP_RATE="0.58"
    STEP_SCALE="1.5"
    
    PRUNE_EPOCH_NO=50
    pruning_ratio_pretrain_cifar10_resnet prune $NET $PRUNE_DIR/$PRUNE_EPOCH_NO.$NET.1.$FLOP_RATE $FLOP_RATE $PRUNE_EPOCH_NO $STEP_SCALE
    pruning_ratio_pretrain_cifar10_resnet prune $NET $PRUNE_DIR/$PRUNE_EPOCH_NO.$NET.2.$FLOP_RATE $FLOP_RATE $PRUNE_EPOCH_NO $STEP_SCALE
    pruning_ratio_pretrain_cifar10_resnet prune $NET $PRUNE_DIR/$PRUNE_EPOCH_NO.$NET.3.$FLOP_RATE $FLOP_RATE $PRUNE_EPOCH_NO $STEP_SCALE
}

run_56()
{
    NET="resnet56"
    FLOP_RATE="0.55"
    
    PRUNE_EPOCH_NO=50
    pruning_ratio_pretrain_cifar10_resnet prune $NET $PRUNE_DIR/$PRUNE_EPOCH_NO.$NET.1.$FLOP_RATE $FLOP_RATE $PRUNE_EPOCH_NO
    pruning_ratio_pretrain_cifar10_resnet prune $NET $PRUNE_DIR/$PRUNE_EPOCH_NO.$NET.2.$FLOP_RATE $FLOP_RATE $PRUNE_EPOCH_NO
    pruning_ratio_pretrain_cifar10_resnet prune $NET $PRUNE_DIR/$PRUNE_EPOCH_NO.$NET.3.$FLOP_RATE $FLOP_RATE $PRUNE_EPOCH_NO
}

run_32()
{
    NET="resnet32"
    FLOP_RATE="0.53"
    STEP_SCALE="1.0"

    PRUNE_EPOCH_NO=50
    pruning_ratio_pretrain_cifar10_resnet prune $NET $PRUNE_DIR/$PRUNE_EPOCH_NO.$NET.1.$FLOP_RATE.$STEP_SCALE $FLOP_RATE $PRUNE_EPOCH_NO $STEP_SCALE
    pruning_ratio_pretrain_cifar10_resnet prune $NET $PRUNE_DIR/$PRUNE_EPOCH_NO.$NET.2.$FLOP_RATE.$STEP_SCALE $FLOP_RATE $PRUNE_EPOCH_NO $STEP_SCALE
    pruning_ratio_pretrain_cifar10_resnet prune $NET $PRUNE_DIR/$PRUNE_EPOCH_NO.$NET.3.$FLOP_RATE.$STEP_SCALE $FLOP_RATE $PRUNE_EPOCH_NO $STEP_SCALE
}

run_20()
{
    NET="resnet20"
    FLOP_RATE="0.50"
    PRUNE_EPOCH_NO=50

    STEP_SCALE="1.5"
    pruning_ratio_pretrain_cifar10_resnet prune $NET $PRUNE_DIR/$PRUNE_EPOCH_NO.$NET.1.$FLOP_RATE.$STEP_SCALE $FLOP_RATE $PRUNE_EPOCH_NO $STEP_SCALE
    pruning_ratio_pretrain_cifar10_resnet prune $NET $PRUNE_DIR/$PRUNE_EPOCH_NO.$NET.2.$FLOP_RATE.$STEP_SCALE $FLOP_RATE $PRUNE_EPOCH_NO $STEP_SCALE
    pruning_ratio_pretrain_cifar10_resnet prune $NET $PRUNE_DIR/$PRUNE_EPOCH_NO.$NET.3.$FLOP_RATE.$STEP_SCALE $FLOP_RATE $PRUNE_EPOCH_NO $STEP_SCALE    
}

run_20
run_32
run_56
run_110