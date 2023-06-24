
EVAL_DIR="resnet_cifar10_eval"

eval_cifar10_resnet()
{
    python main_resnet_cifar10.py --data_path ./data/cifar.python --dataset cifar10 \
    --mode $1 \
    --arch $2 \
    --save_path $3 \
    --baseline_path $4 \
    --pruned_path $5 
}

NET="" # provide resnet name eg. resnet110
BASELINE_PATH="" # give baseline path
PRUNED_PATH="" # give pruned path

eval_cifar10_resnet eval $NET $EVAL_DIR/$NET $BASELINE_PATH $PRUNED_PATH
