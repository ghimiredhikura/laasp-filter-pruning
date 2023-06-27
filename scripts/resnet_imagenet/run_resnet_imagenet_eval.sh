
IMAGENET_PATH="C:/ImageNet"
EVAL_DIR="ImageNet_EVAL"

eval_resnet_imagenet()
{
    python main_resnet_imagenet.py --data_path $IMAGENET_PATH \
    --mode $1 \
    --arch $2 \
    --save_path $3 \
    --baseline_path $4 \
    --pruned_path $5 \
    --workers 8 \
    --batch_size 256
}

# resnet50 
BASELINE_PATH="models_baseline/ImageNet/ResNet50/resnet50.model_best.pth.tar"
PRUNED_PATH="models_pruned\ImageNet\ResNet50_FlopRed_0.54\resnet50.model_best_small.pth.tar"
eval_resnet_imagenet eval resnet50 $EVAL_DIR/resnet50_0.54 $BASELINE_PATH $PRUNED_PATH