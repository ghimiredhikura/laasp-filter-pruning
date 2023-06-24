
# https://xcdskd.readthedocs.io/en/latest/cross_correlation/cross_correlation_coefficient.html

import matplotlib.pyplot as plt
import numpy as np
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2

import torchvision
from torch.autograd import Variable

import time

def norm_data(data):
    """
    normalize data to have mean=0 and standard_deviation=1
    """
    eps=1e-8
    mean_data=np.mean(data)
    std_data=np.std(data, ddof=1)
    return (data-mean_data)/(std_data + eps)

def ncc(data0, data1):
    """
    normalized cross-correlation coefficient between two data sets

    Parameters
    ----------
    data0, data1 :  numpy arrays of same size
    """
    return (1.0/(data0.size-1)) * np.sum(norm_data(data0)*norm_data(data1))

def get_ncc_sim_matrix(data):
    length = data.shape[0]  
    nccv = np.zeros((length, length))
    for i in range(length-1):
        for j in range(i+1, length):
            nccv[i][j] = ncc(data[i], data[j])
            nccv[j][i] = nccv[i][j]
    return nccv

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class RecorderMeter(object):
  """Computes and stores the minimum loss value and its epoch index"""
  def __init__(self, total_epoch):
    self.reset(total_epoch)

  def reset(self, total_epoch):
    assert total_epoch > 0
    self.total_epoch   = total_epoch
    self.current_epoch = 0
    self.epoch_losses  = np.zeros((self.total_epoch, 2), dtype=np.float32) # [epoch, train/val]
    self.epoch_losses  = self.epoch_losses - 1

    self.epoch_accuracy= np.zeros((self.total_epoch, 2), dtype=np.float32) # [epoch, train/val]
    self.epoch_accuracy= self.epoch_accuracy

  def update(self, idx, train_loss, train_acc, val_loss, val_acc):
    assert idx >= 0 and idx < self.total_epoch, 'total_epoch : {} , but update with the {} index'.format(self.total_epoch, idx)
    self.epoch_losses  [idx, 0] = train_loss
    self.epoch_losses  [idx, 1] = val_loss
    self.epoch_accuracy[idx, 0] = train_acc
    self.epoch_accuracy[idx, 1] = val_acc
    self.current_epoch = idx + 1
    
    return self.max_accuracy(False) == val_acc

  def max_accuracy(self, istrain):
    if self.current_epoch <= 0: return 0
    if istrain: return self.epoch_accuracy[:self.current_epoch, 0].max()
    else:       return self.epoch_accuracy[:self.current_epoch, 1].max()
  
  def plot_curve(self, save_path):
    title = 'the accuracy/loss curve of train/val'
    dpi = 80  
    width, height = 1200, 800
    legend_fontsize = 10
    scale_distance = 48.8
    figsize = width / float(dpi), height / float(dpi)

    fig = plt.figure(figsize=figsize)
    x_axis = np.array([i for i in range(self.total_epoch)]) # epochs
    y_axis = np.zeros(self.total_epoch)

    plt.xlim(0, self.total_epoch)
    plt.ylim(0, 100)
    interval_y = 5
    interval_x = 5
    plt.xticks(np.arange(0, self.total_epoch + interval_x, interval_x))
    plt.yticks(np.arange(0, 100 + interval_y, interval_y))
    plt.grid()
    plt.title(title, fontsize=20)
    plt.xlabel('the training epoch', fontsize=16)
    plt.ylabel('accuracy', fontsize=16)
  
    y_axis[:] = self.epoch_accuracy[:, 0]
    plt.plot(x_axis, y_axis, color='g', linestyle='-', label='train-accuracy', lw=2)
    plt.legend(loc=4, fontsize=legend_fontsize)

    y_axis[:] = self.epoch_accuracy[:, 1]
    plt.plot(x_axis, y_axis, color='y', linestyle='-', label='valid-accuracy', lw=2)
    plt.legend(loc=4, fontsize=legend_fontsize)

    
    y_axis[:] = self.epoch_losses[:, 0]
    plt.plot(x_axis, y_axis*50, color='g', linestyle=':', label='train-loss-x50', lw=2)
    plt.legend(loc=4, fontsize=legend_fontsize)

    y_axis[:] = self.epoch_losses[:, 1]
    plt.plot(x_axis, y_axis*50, color='y', linestyle=':', label='valid-loss-x50', lw=2)
    plt.legend(loc=4, fontsize=legend_fontsize)

    if save_path is not None:
      fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
      #print ('---- save figure {} into {}'.format(title, save_path))
    plt.close(fig)
    

def time_string():
  ISOTIMEFORMAT='%Y-%m-%d %X'
  string = '[{}]'.format(time.strftime( ISOTIMEFORMAT, time.gmtime(time.time()) ))
  return string

def convert_secs2time(epoch_time):
  need_hour = int(epoch_time / 3600)
  need_mins = int((epoch_time - 3600*need_hour) / 60)
  need_secs = int(epoch_time - 3600*need_hour - 60*need_mins)
  return need_hour, need_mins, need_secs

def time_file_str():
  ISOTIMEFORMAT='%Y-%m-%d'
  string = '{}'.format(time.strftime( ISOTIMEFORMAT, time.gmtime(time.time()) ))
  return string + '-{}'.format(random.randint(1, 10000))

def timing(f):
    def wrap(*args):
        time1 = time.time()
        ret = f(*args)
        time2 = time.time()
        print ('%s function took %0.3f ms' % (f.__name__, (time2-time1)*1000.0))
        return ret
    return wrap

def feature_visualize(model, dataset, data_loader, imagefile):

    model.to('cpu')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs = 1
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(data_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            
            print('Epoch {}, Batch idx {}, loss {}'.format(
                epoch, batch_idx, loss.item()))
            
            break

    def normalize_output(img):
        img = img - img.min()
        img = img / img.max()
        return img

    # Plot some images
    idx = torch.randint(0, output.size(0), ())
    pred = normalize_output(output[idx, 0])
    img = data[idx, 0]

    # Visualize feature maps
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

    model.conv1.register_forward_hook(get_activation('conv1'))
    data, _ = dataset[0]

    im = cv2.imread(imagefile)
    im = cv2.resize(im, (32,32)).transpose(2, 0, 1)
    data = torch.from_numpy(im)
    data = data.float()

    data.unsqueeze_(0)
    output = model(data)
    act = activation['conv1'].squeeze()

    plt.figure(figsize=(20,20))    
    for idx in range(act.size(0)):
        plt.subplot(8, 8, idx+1)
        plt.imshow(act[idx])
        plt.axis("off")

    plt.tight_layout()
    plt.show()

# refer to: https://github.com/Eric-mingjie/rethinking-network-pruning/blob/master/imagenet/l1-norm-pruning/compute_flops.py
def get_n_params(model):
    total = sum([param.nelement() if param.requires_grad else 0 for param in model.parameters()])
    total /= 1e6
    return total

# The above 'get_n_params' requires 'param.requires_grad' to be true. In KD, for the teacher, this is not the case.
def get_n_params_(model):
    n_params = 0
    for _, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear): # only consider Conv2d and Linear, no BN
            n_params += module.weight.numel()
            if hasattr(module, 'bias') and type(module.bias) != type(None):
                n_params += module.bias.numel()
    return n_params

def get_n_flops(model=None, input_res=224, multiply_adds=True, n_channel=3):
    model = copy.deepcopy(model)

    prods = {}
    def save_hook(name):
        def hook_per(self, input, output):
            prods[name] = np.prod(input[0].shape)
        return hook_per

    list_1=[]
    def simple_hook(self, input, output):
        list_1.append(np.prod(input[0].shape))
    list_2={}
    def simple_hook2(self, input, output):
        list_2['names'] = np.prod(input[0].shape)


    list_conv=[]
    def conv_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        kernel_ops = self.kernel_size[0] * self.kernel_size[1] * (self.in_channels / self.groups)
        bias_ops = 0 if self.bias is not None else 0

        # params = output_channels * (kernel_ops + bias_ops) # @mst: commented since not used
        # flops = (kernel_ops * (2 if multiply_adds else 1) + bias_ops) * output_channels * output_height * output_width * batch_size

        num_weight_params = (self.weight.data != 0).float().sum() # @mst: this should be considering the pruned model
        # could be problematic if some weights happen to be 0.
        flops = (num_weight_params * (2 if multiply_adds else 1) + bias_ops * output_channels) * output_height * output_width * batch_size

        list_conv.append(flops)

    list_linear=[]
    def linear_hook(self, input, output):
        batch_size = input[0].size(0) if input[0].dim() == 2 else 1

        weight_ops = self.weight.nelement() * (2 if multiply_adds else 1)
        bias_ops = self.bias.nelement()

        flops = batch_size * (weight_ops + bias_ops)
        list_linear.append(flops)

    list_bn=[]
    def bn_hook(self, input, output):
        list_bn.append(input[0].nelement() * 2)

    list_relu=[]
    def relu_hook(self, input, output):
        list_relu.append(input[0].nelement())

    list_pooling=[]
    def pooling_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        kernel_ops = self.kernel_size * self.kernel_size
        bias_ops = 0
        params = 0
        flops = (kernel_ops + bias_ops) * output_channels * output_height * output_width * batch_size

        list_pooling.append(flops)

    list_upsample=[]

    # For bilinear upsample
    def upsample_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        flops = output_height * output_width * output_channels * batch_size * 12
        list_upsample.append(flops)

    def foo(net):
        childrens = list(net.children())
        if not childrens:
            if isinstance(net, torch.nn.Conv2d):
                net.register_forward_hook(conv_hook)
            if isinstance(net, torch.nn.Linear):
                net.register_forward_hook(linear_hook)
            if isinstance(net, torch.nn.BatchNorm2d):
                net.register_forward_hook(bn_hook)
            if isinstance(net, torch.nn.ReLU):
                net.register_forward_hook(relu_hook)
            if isinstance(net, torch.nn.MaxPool2d) or isinstance(net, torch.nn.AvgPool2d):
                net.register_forward_hook(pooling_hook)
            if isinstance(net, torch.nn.Upsample):
                net.register_forward_hook(upsample_hook)
            return
        for c in childrens:
            foo(c)

    if model == None:
        model = torchvision.models.alexnet()
    foo(model)
    input = Variable(torch.rand(n_channel,input_res,input_res).unsqueeze(0), requires_grad = True)
    out = model(input)


    total_flops = (sum(list_conv) + sum(list_linear)) # + sum(list_bn) + sum(list_relu) + sum(list_pooling) + sum(list_upsample))
    total_flops /= 1e9
    # print('  Number of FLOPs: %.2fG' % total_flops)

    return total_flops

# The above version is redundant. Get a neat version as follow.
def get_n_flops_(model=None, img_size=(224,224), n_channel=3, count_adds=True, input=None, **kwargs):
    '''Only count the FLOPs of conv and linear layers (no BN layers etc.). 
    Only count the weight computation (bias not included since it is negligible)
    '''
    if hasattr(img_size, '__len__'):
        height, width = img_size
    else:
        assert isinstance(img_size, int)
        height, width = img_size, img_size

    # model = copy.deepcopy(model)
    list_conv = []
    def conv_hook(self, input, output):
        flops = np.prod(self.weight.data.shape) * output.size(2) * output.size(3) / self.groups
        list_conv.append(flops)

    list_linear = []
    def linear_hook(self, input, output):
        flops = np.prod(self.weight.data.shape)
        list_linear.append(flops)

    def register_hooks(net, hooks):
        childrens = list(net.children())
        if not childrens:
            if isinstance(net, torch.nn.Conv2d):
                h = net.register_forward_hook(conv_hook)
                hooks += [h]
            if isinstance(net, torch.nn.Linear):
                h = net.register_forward_hook(linear_hook)
                hooks += [h]
            return
        
        for c in childrens:
            register_hooks(c, hooks)
    
    hooks = []
    register_hooks(model, hooks)
    if input is None:
        input = torch.rand(1, n_channel, height, width)
        use_cuda = next(model.parameters()).is_cuda
        if use_cuda:
            input = input.cuda()
    
    # forward
    is_train = model.training 
    model.eval()
    with torch.no_grad():
        model(input, **kwargs)
    total_flops = (sum(list_conv) + sum(list_linear))
    if count_adds:
        total_flops *= 2
    
    # reset to original model
    for h in hooks: h.remove() # clear hooks
    if is_train: model.train()
    return total_flops

def count_conv_filters(model):
    model.cpu()
    total_filters = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            weight = m.weight.detach().numpy()
            total_filters+= float(weight.shape[0])
    return total_filters

def test1d():
    ndata=50
    U_true= 3 *(-1.0 + 2.0*np.random.rand(ndata)) # vary between -3 and 3
    print('U_true:')
    print(U_true)
    V_true=2.8*U_true+9.2 # V depends linearly on the *random* U
    print('V_true:')
    print(V_true)
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    ax.plot(U_true, label='U')
    ax.plot(V_true, label='V')
    ax.legend(loc='upper left', shadow=True, fancybox=True)
    ax.set_ylabel('data values')
    ax.set_xlabel('observations')
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(U_true,V_true, label= "V vs. U, perfectly correlated")
    ax.legend(loc='upper right', shadow=True, fancybox=True)
    ax.set_ylabel('V values')
    ax.set_xlabel('U values')
    ax.set_title('linear correlation between U and V random variables')
    plt.show()

    nccv=ncc(U_true,V_true)
    print("NCC: ", nccv)

if __name__ == '__main__':
    test1d()