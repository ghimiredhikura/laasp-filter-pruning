EVAL_DIR="vgg16_cifar10_eval"

eval_cifar10_vgg16()
{
    python main_vgg_cifar10.py --data_path ./data/cifar.python --dataset cifar10 \
    --depth 16 \
    --mode $1 \
    --save_path $EVAL_DIR \
    --pruned_path $2 \
    --baseline_path $3
}

BASELINE_PATH="none"
for ROUND in 1 2 3
do
    PRUNED_PATH="../Models/pruned/CIFAR10/Vgg16_FLOPs_RR_60.50/model"$ROUND"_0.60/model_best.pth.tar" # replace with path in your directory. 
    eval_cifar10_vgg16 eval $PRUNED_PATH $BASELINE_PATH
done