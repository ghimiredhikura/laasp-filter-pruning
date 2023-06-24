
PRUNE_PATH="vgg_cifar10_prune"

prune_baseline_vgg16()
{
    python main_vgg_cifar10.py --dataset cifar10 --depth 16 \
    --mode prune \
    --save_path $1 \
    --rate_flop $2 \
    --total_epoches 160 \
    --schedule 40 80 120 \
    --gammas 0.2 0.2 0.2 \
    --recover_epoch 1 \
    --prune_epoch 30 \
    --lr 0.1 --decay 0.0005 --batch_size 256
}

prune_baseline_vgg16 $PRUNE_PATH/model1_0.342 0.342
prune_baseline_vgg16 $PRUNE_PATH/model2_0.342 0.342
prune_baseline_vgg16 $PRUNE_PATH/model3_0.342 0.342