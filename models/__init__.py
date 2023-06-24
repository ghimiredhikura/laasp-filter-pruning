
# imagenet based resnet
from .imagenet_resnet import resnet18, resnet34, resnet50, resnet101, resnet152

# cifar10 based resnet 
from .resnet_cifar import resnet20, resnet32, resnet44, resnet56, resnet110

from torchvision.models import mobilenet_v2