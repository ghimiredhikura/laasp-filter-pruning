
PRUNE_DIR="cifar10_prune"

pruning_ratio_pretrain_cifar10_resnet()
{
    python main_resnet_cifar10.py.py --data_path ./data/cifar.python --dataset cifar10 \
    --mode $1 \
    --arch $2 \
    --save_path $3 \
    --rate_flop $4 \
    --total_epoches 160 \
    --schedule 40 80 120 \
    --gammas 0.2 0.2 0.2 \
    --recover_epoch 1 \
    --prune_epoch $5 \
    --workers 8 \
    --lr 0.1 --decay 0.0005 --batch_size 256
}

run_20()
{
    NET="resnet20"
    FLOP_RATE="0.300"
    PRUNE_EPOCH_NO=$1
    ROUND=$2
    pruning_ratio_pretrain_cifar10_resnet prune $NET $PRUNE_DIR/$PRUNE_EPOCH_NO.$NET.$ROUND.$FLOP_RATE $FLOP_RATE $PRUNE_EPOCH_NO
}

for ROUND in 1 2 3
do
    for PRUNE_EPOCH_NO in 10 20 30 40 50 60 70 80 90 100 110 120 130
    do 
        run_20 $PRUNE_EPOCH_NO $ROUND
    done
done