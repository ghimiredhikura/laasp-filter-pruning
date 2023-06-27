
import sys, os, random, shutil, time
import copy
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import torch_pruning_tool.torch_pruning as tp
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.datasets as dset
import torchvision.transforms as transforms
import numpy as np
from utils.utils import AverageMeter, RecorderMeter, time_string
from utils.utils import convert_secs2time, get_ncc_sim_matrix, get_n_flops_, get_n_params_

import models

from scipy.spatial import distance

parser = argparse.ArgumentParser()

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser.add_argument("--data_path", type=str, help='Path to dataset')
parser.add_argument('--baseline_path', default='./', type=str, help='..path of baseline model')
parser.add_argument('--pretrain_path', default='./', type=str, help='..path of pre-trained model')
parser.add_argument('--pruned_path', default='./', type=str, help='..path of pruned model')
parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'imagenet', 'svhn', 'stl10'],
                    help='Choose between Cifar10/100 and ImageNet.')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet20',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet20)')
parser.add_argument('--save_path', type=str, default='./', help='Folder to save checkpoints and log.')
parser.add_argument('--mode', type=str, default="eval", choices=['train', 'eval', 'prune'])
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--verbose', action='store_true', default=False)
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--prune_epoch', type=int, default=30)
parser.add_argument('--total_epoches', type=int, default=160)
parser.add_argument('--recover_epoch', type=int, default=1)
parser.add_argument('--lr', type=float, default=0.01, metavar='LR', help='learning rate (default: 0.1)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
parser.add_argument('--decay', '--wd', default=0.0005, type=float, metavar='W', help='weight decay (default: 0.0005)')
parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
                    help='Decrease learning rate at these epochs.')
parser.add_argument('--gammas', type=float, nargs='+', default=[0.1, 0.1],
                    help='LR is multiplied by gamma on schedule, number of gammas should be equal to schedule')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')

# Acceleration
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--workers', type=int, default=4, help='number of data loading workers (default: 2)')

# compress rate
parser.add_argument('--rate_flop', type=float, default=0.342, help='This is flop reduction rate')

parser.add_argument('--max_prune_limit', type=float, default=0.65, help='This is flop reduction rate')
parser.add_argument('--step_scale', type=float, default=0.01, help='This is flop reduction rate')

# recover flop rate 
parser.add_argument('--recover_flop', type=float, default=0.00, help='This is flop reduction rate')

# random seed
parser.add_argument('--manualSeed', type=int, help='manual seed')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device('cuda:0' if args.cuda else 'cpu')
args.use_cuda = args.ngpu > 0 and torch.cuda.is_available()

if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if args.use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)
cudnn.benchmark = True

def print_log(print_string, log, display=True):
    if display:
        print("{}".format(print_string))
    log.write('{}\n'.format(print_string))
    log.flush()

def main():

    # Init logger
    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)

    log = open(os.path.join(args.save_path, 'log_seed_{}.txt'.format(args.manualSeed)), 'w')

    print_log('save path : {}'.format(args.save_path), log)
    state = {k: v for k, v in args._get_kwargs()}
    print_log(state, log)
    print_log("python version : {}".format(sys.version.replace('\n', ' ')), log)
    print_log("torch  version : {}".format(torch.__version__), log)
    print_log("cudnn  version : {}".format(torch.backends.cudnn.version()), log)
    print_log("Pretrain path: {}".format(args.pretrain_path), log)
    print_log("Pruned path: {}".format(args.pruned_path), log)

    # Init dataset
    if not os.path.isdir(args.data_path):
        os.makedirs(args.data_path)

    if args.dataset == 'cifar10':
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
    elif args.dataset == 'cifar100':
        mean = [x / 255 for x in [129.3, 124.1, 112.4]]
        std = [x / 255 for x in [68.2, 65.4, 70.4]]
    else:
        assert False, "Unknow dataset : {}".format(args.dataset)

    train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4), transforms.ToTensor(),
         transforms.Normalize(mean, std)])
    test_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean, std)])

    if args.dataset == 'cifar10':
        train_data = dset.CIFAR10(args.data_path, train=True, transform=train_transform, download=True)
        test_data = dset.CIFAR10(args.data_path, train=False, transform=test_transform, download=True)
        num_classes = 10
    elif args.dataset == 'cifar100':
        train_data = dset.CIFAR100(args.data_path, train=True, transform=train_transform, download=True)
        test_data = dset.CIFAR100(args.data_path, train=False, transform=test_transform, download=True)
        num_classes = 100
    else:
        assert False, 'Do not support dataset : {}'.format(args.dataset)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.workers)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.workers)

    # subset of train dataset, 5000
    subset_index = random.sample(range(0, 49999), 10000)
    dataset_subset = torch.utils.data.Subset(train_data, subset_index)
    loader_subset = torch.utils.data.DataLoader(dataset_subset, batch_size=256, shuffle=False, 
                                              num_workers=args.workers)

    # create model
    print_log("=> creating model '{}'".format(args.arch), log)
    # Init model, criterion, and optimizer
    model = models.__dict__[args.arch](num_classes)
    print_log("=> Model : {}".format(model), log, True)
    print_log("=> parameter : {}".format(args), log)

    # define loss function (criterion) and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    if args.use_cuda:
        criterion.cuda()

    if args.mode == 'prune':

        if os.path.isfile(args.pretrain_path):

            print("Loading Model State Dict from: ", args.pretrain_path)
            pretrain = torch.load(args.pretrain_path)
            model = pretrain['state_dict']
        else:
            print("No pretrain model is given, pruning from scrach")

        if args.use_cuda:
            torch.cuda.empty_cache()
            model.cuda()

        # train for 0 ~ args.prune_epoch
        train(model, train_loader, test_loader, criterion, args.start_epoch, args.prune_epoch, log)
        
        # save halfway trained model just before pruning 
        save_checkpoint({
            'epoch': 0,
            'arch': args.arch,
            'state_dict': model,
        }, False, args.save_path, '%s.epoch.%d.pth.tar' % (args.arch, args.prune_epoch) )

        pruned_model = prune(model, train_loader, loader_subset, test_loader, criterion, log)
        print_log("=> Model [After Pruning]:\n {}".format(pruned_model), log)

        # train for remaining epoch: args.prune_epoch ~ args.total_epoches 
        train(pruned_model, train_loader, test_loader, criterion, args.prune_epoch, args.total_epoches, log)        

    if args.mode == 'train':

        if os.path.isfile(args.pretrain_path):
            print("Loading Model State Dict from: ", args.pretrain_path)
            pretrain = torch.load(args.pretrain_path)
            model = pretrain['state_dict']
        else:
            print("Training model from init.. ")

        print_log("=> train network :\n {}".format(model), log, True)

        if args.use_cuda:
            torch.cuda.empty_cache()
            model.cuda()

        train(model, train_loader, test_loader, criterion, args.start_epoch, args.total_epoches, log)

    elif args.mode == 'eval':       

        base_size = get_n_params_(model)
        base_ops = get_n_flops_(model, img_size=(32, 32))

        if os.path.isfile(args.baseline_path):
            print("Loading Baseline Model from: ", args.baseline_path)
            baseline = torch.load(args.baseline_path)
            model = baseline['state_dict']
            print_log("=> Baseline network :\n {}".format(model), log, False)

            if args.use_cuda:
                torch.cuda.empty_cache()
                model.cuda()

            val_acc_top1, val_acc_top5, val_loss = validate(test_loader, model, criterion)
            print_log("Baseline Val Acc@1: %0.3lf, Acc@5: %0.3lf,  Loss: %0.5f" % (val_acc_top1, val_acc_top5, val_loss), log)

        if os.path.isfile(args.pruned_path):
            print("Load Pruned Model from %s" % (args.pruned_path))
            pruned = torch.load(args.pruned_path)
            model_pruned = pruned['state_dict']
            print_log("=> pruned network :\n {}".format(model_pruned), log, True)
            
            if args.use_cuda:
                torch.cuda.empty_cache()
                model_pruned.cuda()

            pruned_size = get_n_params_(model_pruned)
            pruned_ops = get_n_flops_(model_pruned, img_size=(32, 32))
            val_acc_top1, val_acc_top5, val_loss = validate(test_loader, model_pruned, criterion)
            print_log("Pruned Val Acc@1: %0.3lf, Acc@5: %0.3lf,  Loss: %0.5f" % (val_acc_top1, val_acc_top5, val_loss), log)

            print_log(
            "Params: {:.2f} M => {:.2f} M, (Param RR {:.2f}%)".format(
                base_size / 1e6, pruned_size / 1e6, (1.0 - pruned_size / base_size) * 100 ), log)
            print_log(
            "FLOPs: {:.2f} M => {:.2f} M (FLOPs RR {:.2f}%, Speed-Up {:.2f}X )".format(
                base_ops / 1e6,
                pruned_ops / 1e6,
                (1.0 - pruned_ops / base_ops) * 100,
                base_ops / pruned_ops ), log)

    log.close()

def prune(model, train_loader, loader_subset, test_loader, criterion, log):

    sub_inputs, sub_targets = get_train_subset_in_memory(loader_subset)

    with torch.no_grad():
        val_acc_top1, val_acc_top5, val_loss = validate(test_loader, model, criterion)
        print_log("Val Acc@1: %0.3lf, Acc@5: %0.3lf,  Loss: %0.5f" % (val_acc_top1, val_acc_top5, val_loss), log) 

    flops_baseline = get_n_flops_(model, img_size=(32, 32))

    current_flop_reduction_rate = 0.0
    flop_reduction_rate_temp = 0.0
    filter_prune_limit_per_layer = args.max_prune_limit

    model.eval()

    conv_filter_prune_idx_all = get_all_conv_index_to_prune(model)
    print("conv_filter_prune_idx_all", conv_filter_prune_idx_all)
    layerwise_filter_count_org = get_conv_filter_count_v1(model, conv_filter_prune_idx_all)
    print("layerwise_filter_count_org", layerwise_filter_count_org)
    print("Start Pruning ...")
    methods = ['l1norm', 'l2norm', 'eucl', 'cos']

    #methods = ['l1norm', 'eucl']
    prune_stat = {}
    for m in methods:
        prune_stat[m] = 0

    layers_step_prune_count = calc_conv_layers_step_prune_count(model, conv_filter_prune_idx_all, args.step_scale)
    print("layers_step_prune_count", layers_step_prune_count)

    while current_flop_reduction_rate < args.rate_flop:
        small_loss = 100000000000.0
        small_loss_lindex = 0
        opt_method = ''

        for prune_conv_idx in conv_filter_prune_idx_all:
            for method in methods:
                if layers_step_prune_count[prune_conv_idx] <= 0:
                    continue
                # model copy to prune 
                model_copy = copy.deepcopy(model)
                # prune 
                model_copy.eval()
                prune_flag = prune_single_conv_layer_v1(model_copy, prune_conv_idx, layerwise_filter_count_org[prune_conv_idx], 
                                        layers_step_prune_count[prune_conv_idx], filter_prune_limit_per_layer, method)
                if prune_flag == False:
                    continue

                # calc loss after prune 
                if args.use_cuda:
                    torch.cuda.empty_cache()
                    model_copy.cuda()
                with torch.no_grad():
                    _, sample_loss = validate_fast(sub_inputs, sub_targets, model_copy, criterion)

                # store conv layer index with small loss 
                if sample_loss < small_loss:
                    small_loss_lindex = prune_conv_idx
                    small_loss = sample_loss
                    opt_method = method

        # prune selected layer with given prune rate
        prune_single_conv_layer_v1(model, small_loss_lindex, layerwise_filter_count_org[small_loss_lindex],
                            layers_step_prune_count[small_loss_lindex], filter_prune_limit_per_layer, opt_method)

        prune_stat[opt_method] += layers_step_prune_count[small_loss_lindex]

        flops_pruned = get_n_flops_(model, img_size=(32, 32))
        current_flop_reduction_rate = 1.0 - flops_pruned / flops_baseline
        print_log("[Pruning Method: %s] Flop Reduction Rate: %lf/%lf [Pruned %d filters from %s]" % (opt_method,
                    current_flop_reduction_rate, args.rate_flop, layers_step_prune_count[small_loss_lindex], small_loss_lindex), log)

        if current_flop_reduction_rate - flop_reduction_rate_temp >= args.recover_flop:
            if args.use_cuda:
                torch.cuda.empty_cache()
                model.cuda()
            # train 
            train(model, train_loader, test_loader, criterion, args.prune_epoch, args.prune_epoch  + args.recover_epoch, log)
            flop_reduction_rate_temp = current_flop_reduction_rate

    print_log('Prune Stats: ' + ''.join(str(prune_stat)), log)

    layerwise_filter_count_prune = get_conv_filter_count_v1(model, conv_filter_prune_idx_all)
    print_log('Final Flop Reduction Rate: %.4lf' % current_flop_reduction_rate, log)
    print_log("Conv Filters Before Pruning: " + ''.join(str(layerwise_filter_count_org)), log)
    print_log("Conv Filters After Pruning: " + ''.join(str(layerwise_filter_count_prune)), log)    

    filter_prune_rate = {}
    for idx in layerwise_filter_count_org.keys():
        filter_prune_rate[idx] = 1.0 - float(layerwise_filter_count_prune[idx]/float(layerwise_filter_count_org[idx]))
    print_log("Layerwise Pruning Rate: " + ''.join(str(filter_prune_rate)), log)    

    # save pruned model before finetuning
    save_checkpoint( {
        'epoch': 0,
        'arch': args.arch,
        'state_dict': model,
    }, False, args.save_path, '%s.init.prune.pth.tar' % (args.arch) )

    return model

def get_all_conv_index_to_prune(model):
    conv_filter_prune_idx = []
    for idx, m in enumerate(model.modules()):
        if not isinstance(m, nn.Conv2d): # check if it is conv layer 
            continue
        conv_filter_prune_idx.append(idx)
    return conv_filter_prune_idx

# this function will return number of filters to be pruned from each layer 
# that results in ~1% reduction in FLOPs
def calc_conv_layers_step_prune_count(model, conv_filter_prune_idx, step_scale=1.0):
    layers_step_prune_count = {}
    for id in conv_filter_prune_idx:  
        model_org = copy.deepcopy(model)
        input = torch.randn(1,3,32,32)
        DG = tp.DependencyGraph().build_dependency(model_org, input.cuda())
        flops_baseline = get_n_flops_(model_org, img_size=(32, 32))  
        for idx, m in enumerate(model_org.modules()):
            if isinstance(m, nn.Conv2d) and idx == id:
                pruning_idxs = norm_prune(m, 1)
                plan = DG.get_pruning_plan(m, tp.prune_conv_out_channel, pruning_idxs)
                plan.exec() 
                break
        flops_pruned = get_n_flops_(model_org, img_size=(32, 32))
        filter_prune_step = int(step_scale / ((1.0 - flops_pruned / flops_baseline) * 100) + 0.5)

        layers_step_prune_count[idx] = max(1, filter_prune_step)
        #layers_step_prune_count[idx] = filter_prune_step
    
    return layers_step_prune_count

def get_conv_filter_count_v1(model, conv_filter_prune_idx):
    conv_filter_count = {}    
    for idx, m in enumerate(model.modules()):
        if idx in conv_filter_prune_idx:
            conv_filter_count[idx] = m.weight.shape[0]
    return conv_filter_count

def prune_single_conv_layer_v1(model, conv_id, org_filter_count, step_prune_count=1, 
                            max_pruning_rate=0.7, method='l1norm'):

    max_prune_count = org_filter_count * max_pruning_rate
    prune_flag = False
    DG = tp.DependencyGraph().build_dependency(model, torch.randn(1,3,32,32).cuda())
    for idx, m in enumerate(model.modules()):
        if isinstance(m, nn.Conv2d) and idx == conv_id:
            nfilters = m.weight.shape[0]
            npruned_old = org_filter_count - nfilters
            step_prune_count_new = int(min(step_prune_count, max_prune_count-npruned_old))
            if step_prune_count_new <= 0:
                continue
            prune_conv(DG, m, step_prune_count_new, method)
            prune_flag = True
            break
    return prune_flag

def get_train_subset_in_memory(train_loader_subset):
    sub_inputs = []
    sub_targets = []
    for _, (input, target) in enumerate(train_loader_subset):
        if args.use_cuda:
            target = target.cuda(non_blocking=True)
            input = input.cuda(non_blocking=True)
        sub_targets.append(target)
        sub_inputs.append(input)
    return sub_inputs, sub_targets

def get_similar_matrix_cuda(weight_vec_after_norm, dist_type="eucl"):
    if dist_type == "eucl":
        similar_matrix = torch.cdist(weight_vec_after_norm, weight_vec_after_norm, p=2)
    elif dist_type == "cos":
        weight_vec_after_norm = F.normalize(weight_vec_after_norm, p=2, dim=1)
        similar_matrix = torch.matmul(weight_vec_after_norm, weight_vec_after_norm.t())
        similar_matrix = 1 - similar_matrix
        similar_matrix[torch.isnan(similar_matrix)] = 1
    return similar_matrix

def similarity_prune_cuda(conv, amount=0.2, method='eucl'):
    weight = conv.weight.detach().clone().cuda()
    total_filters = weight.shape[0]
    weight = weight.view(total_filters, -1)
    num_pruned = int(total_filters * amount) if amount < 1.0 else amount

    similar_matrix = get_similar_matrix_cuda(weight, method)
    similar_sum = torch.sum(similar_matrix, dim=1)
    _, pruning_index = torch.sort(similar_sum)
    pruning_index = pruning_index[:num_pruned].tolist()

    return pruning_index

def norm_prune(conv, amount=1, ltype='l1norm'):
    if ltype == 'l1norm':
        strategy = tp.strategy.L1Strategy()
    else:
        strategy = tp.strategy.L2Strategy()
    pruning_index = strategy(conv.weight, amount=amount)
    return pruning_index

def prune_conv(DG, conv, amount, method):
    # get index of filters to be pruned        
    if 'norm' in method:
        pruning_index = norm_prune(conv, amount, method)
        # apply pruning
        plan = DG.get_pruning_plan(conv, tp.prune_conv_out_channel, pruning_index)
        plan.exec()
    else:
        pruning_index = similarity_prune_cuda(conv, amount, method)
        # apply pruning
        plan = DG.get_pruning_plan(conv, tp.prune_conv_out_channel, pruning_index)
        plan.exec()

def train(model, train_loader, test_loader, criterion, start_epoch, total_epoches, log):

    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum,
            weight_decay=args.decay, nesterov=True)

    if args.use_cuda:
        torch.cuda.empty_cache()
        model.cuda()

    recorder = RecorderMeter(args.total_epoches)
    start_time = time.time()
    epoch_time = AverageMeter()
    best_accuracy = 0

    # Main loop
    for epoch in range(start_epoch, total_epoches):

        current_learning_rate = adjust_learning_rate(optimizer, epoch, args.gammas, args.schedule)

        need_hour, need_mins, need_secs = convert_secs2time(epoch_time.avg * (args.total_epoches - epoch))
        need_time = '[Need: {:02d}:{:02d}:{:02d}]'.format(need_hour, need_mins, need_secs)

        # train for one epoch
        train_acc, train_los = train_epoch(train_loader, model, criterion, optimizer, epoch, log)

        # validate
        val_acc_top1, val_acc_top5, val_los = validate(test_loader, model, criterion)
        print_log("Epoch %d/%d [learning_rate=%lf] Val [Acc@1=%0.3f, Acc@5=%0.3f | Loss= %0.5f" % 
                (epoch, args.total_epoches, current_learning_rate, val_acc_top1, val_acc_top5, val_los), log)

        is_best = recorder.update(epoch, train_los, train_acc, val_los, val_acc_top1)
        if recorder.max_accuracy(False) > best_accuracy:
            print_log('\n==>>{:s} [Epoch={:03d}/{:03d}] {:s} [learning_rate={:6.4f}]'.format(time_string(), epoch, args.total_epoches,
                                                                                    need_time, current_learning_rate) \
                + ' [Best : Acc@1={:.2f}, Error={:.2f}]'.format(recorder.max_accuracy(False),
                                                                100 - recorder.max_accuracy(False)), log)
            best_accuracy = recorder.max_accuracy(False)
            is_best = True

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model,
        }, is_best, args.save_path, '%s.checkpoint.pth.tar' % (args.arch))

        # measure elapsed time
        epoch_time.update(time.time() - start_time)
        start_time = time.time()
        recorder.plot_curve(os.path.join(args.save_path, 'curve.png'))

# train function (forward, backward, update)
def train_epoch(train_loader, model, criterion, optimizer, epoch, log):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.use_cuda:
            target = target.cuda(non_blocking=True)
            input = input.cuda(non_blocking=True)

        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1, input.size(0))
        top5.update(prec5, input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()

        # Mask grad for iteration
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    return top1.avg, losses.avg

def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        # Only need to do topk for highest k, reuse for the rest 
        _, pred = output.topk(k=maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        
        batch_size = target.size(0)

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res

def validate_single_epoch(input, target, model, criterion, losses_m, top1_m, top5_m):
    if args.use_cuda:
        target = target.cuda(non_blocking=True)
        input = input.cuda(non_blocking=True)

    output = model(input)
    loss = criterion(output, target)

    # measure accuracy and record loss
    prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

    losses_m.update(loss.item(), input.size(0))
    top1_m.update(prec1, input.size(0))
    top5_m.update(prec5, input.size(0))

    return losses_m, top1_m, top5_m    

def validate(val_loader, model, criterion):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    for i, (input, target) in enumerate(val_loader):
        losses, top1, top5 = validate_single_epoch(input, target, model, criterion, losses, top1, top5)
    return top1.avg, top5.avg, losses.avg

def validate_fast(inputs, targets, model, criterion):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    for (input, target) in zip(inputs, targets):
        losses, top1, top5 = validate_single_epoch(input, target, model, criterion, losses, top1, top5)
    return top1.avg, losses.avg

def save_checkpoint(state, is_best, save_path, filename):
    filename = os.path.join(save_path, filename)
    torch.save(state, filename)
    if is_best:
        bestname = os.path.join(save_path, '%s.model_best.pth.tar' % (args.arch))
        shutil.copyfile(filename, bestname)

def adjust_learning_rate(optimizer, epoch, gammas, schedule):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr
    assert len(gammas) == len(schedule), "length of gammas and schedule should be equal"
    for (gamma, step) in zip(gammas, schedule):
        if (epoch >= step):
            lr = lr * gamma
        else:
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

if __name__=='__main__':
    main()