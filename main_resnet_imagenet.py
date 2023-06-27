
import sys, os, random, shutil, time
import copy
import random
from collections import OrderedDict

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import torch_pruning_tool.torch_pruning as tp
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.datasets as datasets 
from utils.imagenet_utils import presets, transforms, utils, sampler
import numpy as np
from utils.utils import AverageMeter, RecorderMeter, time_string
from utils.utils import convert_secs2time, get_ncc_sim_matrix, get_n_flops_, get_n_params_
from torch.utils.data.dataloader import default_collate
import models
import models.imagenet_resnet as resnet

from scipy.spatial import distance

parser = argparse.ArgumentParser()

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser.add_argument("--data_path", type=str, default="/home/viplab/DataBase/ImageNet2012")
parser.add_argument('--pretrain_path', default='./', type=str, help='..path of pre-trained model')
parser.add_argument('--baseline_path', default='./', type=str, help='..path of baseline model')
parser.add_argument('--pruned_path', default='./', type=str, help='..path of pruned model')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('--save_path', type=str, default='./', help='Folder to save checkpoints and log.')
parser.add_argument('--mode', type=str, default="eval", choices=['train', 'eval', 'prune'])
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--verbose', action='store_true', default=False)
parser.add_argument('--total_epoches', type=int, default=100)
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--prune_epoch', type=int, default=25)
parser.add_argument('--recover_epoch', type=int, default=1)
parser.add_argument('--print-freq', '-p', default=200, type=int, metavar='N', help='print frequency (default: 100)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR', help='learning rate (default: 0.1)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
parser.add_argument('--decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--decay_epoch_step', default=30, type=float, help='weight decay (default: 1e-4)')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument("--dist-url", default="env://", type=str, help="url used to set up distributed training")
parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
parser.add_argument("--mixup-alpha", default=0.0, type=float, help="mixup alpha (default: 0.0)")
parser.add_argument("--cutmix-alpha", default=0.0, type=float, help="cutmix alpha (default: 0.0)")
parser.add_argument("--cache-dataset", dest="cache_dataset", help="Cache the datasets for quicker initialization. It also serializes the transforms", action="store_true")

# Acceleration
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--workers', type=int, default=4, help='number of data loading workers (default: 2)')

# compress rate
parser.add_argument('--rate_flop', type=float, default=0.342, help='This is flop reduction rate')

# random seed
parser.add_argument('--manualSeed', type=int, help='manual seed')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
#device = torch.device('cuda:0' if args.cuda else 'cpu')
device = torch.device(args.device)
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

def _get_cache_path(filepath):
    import hashlib

    h = hashlib.sha1(filepath.encode()).hexdigest()
    cache_path = os.path.join("~", ".torch", "vision", "datasets", "imagefolder", h[:10] + ".pt")
    cache_path = os.path.expanduser(cache_path)
    return cache_path

def load_data(traindir, valdir, args):
    # Data loading code
    print("Loading data...")
    resize_size, crop_size = (342, 299) if args.arch == 'inception_v3' else (256, 224)

    print("Loading training data...")
    st = time.time()
    cache_path = _get_cache_path(traindir)
    if args.cache_dataset and os.path.exists(cache_path):
        # Attention, as the transforms are also cached!
        print("Loading dataset_train from {}".format(cache_path))
        dataset, _ = torch.load(cache_path)
    else:
        auto_augment_policy = getattr(args, "auto_augment", None)
        random_erase_prob = getattr(args, "random_erase", 0.0)
        dataset = torchvision.datasets.ImageFolder(
            traindir,
            presets.ClassificationPresetTrain(crop_size=crop_size, auto_augment_policy=auto_augment_policy,
                                              random_erase_prob=random_erase_prob))
        if args.cache_dataset:
            print("Saving dataset_train to {}...".format(cache_path))
            utils.mkdir(os.path.dirname(cache_path))
            utils.save_on_master((dataset, traindir), cache_path)
    print("Data loading took", time.time() - st)

    print("Loading validation data...")
    cache_path = _get_cache_path(valdir)
    if args.cache_dataset and os.path.exists(cache_path):
        # Attention, as the transforms are also cached!
        print("Loading dataset_test from {}".format(cache_path))
        dataset_test, _ = torch.load(cache_path)
    else:
        dataset_test = torchvision.datasets.ImageFolder(
            valdir,
            presets.ClassificationPresetEval(crop_size=crop_size, resize_size=resize_size))
        if args.cache_dataset:
            print("Saving dataset_test to {}...".format(cache_path))
            utils.mkdir(os.path.dirname(cache_path))
            utils.save_on_master((dataset_test, valdir), cache_path)

    print("Creating data loaders...")
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    return dataset, dataset_test, train_sampler, test_sampler

def main():

    # Init logger
    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)

    init_distributed_mode(args)

    log = open(os.path.join(args.save_path, 'log_seed_{}.txt'.format(args.manualSeed)), 'w')

    print_log("args: {}".format(args), log)

    print_log('save path : {}'.format(args.save_path), log)
    state = {k: v for k, v in args._get_kwargs()}
    print_log(state, log)
    print_log("Random Seed: {}".format(args.manualSeed), log)
    print_log("python version : {}".format(sys.version.replace('\n', ' ')), log)
    print_log("torch  version : {}".format(torch.__version__), log)
    print_log("cudnn  version : {}".format(torch.backends.cudnn.version()), log)
    print_log("Pretrain path: {}".format(args.pretrain_path), log)
    print_log("Pruned path: {}".format(args.pruned_path), log)

    train_dir = os.path.join(args.data_path, "train")
    val_dir = os.path.join(args.data_path, "val")
    dataset, dataset_test, train_sampler, test_sampler = load_data(train_dir, val_dir, args)
    
    collate_fn = None
    num_classes = len(dataset.classes)
    mixup_transforms = []
    if args.mixup_alpha > 0.0:
        mixup_transforms.append(transforms.RandomMixup(num_classes, p=1.0, alpha=args.mixup_alpha))
    if args.cutmix_alpha > 0.0:
        mixup_transforms.append(transforms.RandomCutmix(num_classes, p=1.0, alpha=args.cutmix_alpha))
    if mixup_transforms:
        mixupcutmix = torchvision.transforms.RandomChoice(mixup_transforms)

        def collate_fn(batch):
            return mixupcutmix(*default_collate(batch))

    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size, sampler=test_sampler, num_workers=args.workers, pin_memory=True
    )

    # subset of train dataset, 5000
    subset_index = random.sample(range(0, 149999), 5000)
    dataset_subset = torch.utils.data.Subset(dataset, subset_index)
    loader_subset = torch.utils.data.DataLoader(dataset_subset, batch_size=256, shuffle=False, 
                                              num_workers=args.workers)

    # create model
    print_log("=> creating model '{}'".format(args.arch), log)
    model = models.__dict__[args.arch](pretrained=False)
    print_log("=> Model : {}".format(model), log, False)
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
            args.start_epoch = pretrain['epoch']
        else:
            print("No pretrain model is given, pruning from scrach")

        if args.use_cuda:
            torch.cuda.empty_cache()
            model.cuda()

        # train for 0 ~ args.prune_epoch
        model = train(model, train_loader, test_loader, train_sampler, criterion, args.start_epoch, args.prune_epoch, log)
        # save halfway trained model just before pruning 
        save_checkpoint({
            'epoch': args.prune_epoch + 1,
            'arch': args.arch,
            'state_dict': model,
        }, False, args.save_path, '%s.epoch.%d.pth.tar' % (args.arch, args.prune_epoch) )

        pruned_model = prune(model, train_loader, train_sampler, loader_subset, test_loader, criterion, log)
        print_log("=> Model [After Pruning]:\n {}".format(pruned_model), log)

        # train for remaining epoch: args.prune_epoch ~ args.total_epoches 
        train(pruned_model, train_loader, test_loader, train_sampler, criterion, args.prune_epoch, args.total_epoches, log)

    if args.mode == 'train':

        if os.path.isfile(args.pretrain_path):
            print("Loading Model State Dict from: ", args.pretrain_path)
            pretrain = torch.load(args.pretrain_path)
            model = pretrain['state_dict']
            #args.start_epoch = pretrain['epoch']
        else:
            print("Training model from init.. ")

        print_log("=> train network :\n {}".format(model), log, True)

        if args.use_cuda:
            torch.cuda.empty_cache()
            model.cuda()

        train(model, train_loader, test_loader, train_sampler, criterion, args.start_epoch, args.total_epoches, log)

    elif args.mode == 'eval':       

        base_size = get_n_params_(model)
        base_ops = get_n_flops_(model, img_size=(224, 224))

        if os.path.isfile(args.baseline_path):
            print("Loading Baseline Model from: ", args.baseline_path)
            baseline = torch.load(args.baseline_path)
            model = baseline['state_dict']

            print_log("=> Baseline network :\n {}".format(model), log)
            print_log("Baseline epoch: %d, arch: %s" % (baseline['epoch'], baseline['arch']), log)

            if args.use_cuda:
                torch.cuda.empty_cache()
                model.cuda()

            val_acc_top1, val_acc_top5, val_loss = validate(test_loader, model, criterion)
            print_log("Baseline Val Acc@1: %0.3lf, Acc@5: %0.3lf,  Loss: %0.5f" % (val_acc_top1, val_acc_top5, val_loss), log)            

        if os.path.isfile(args.pruned_path):
            print("Load Pruned Model from %s" % (args.pruned_path))
            pruned = torch.load(args.pruned_path)
            model_pruned = pruned['state_dict']
            print_log("=> pruned network :\n {}".format(model_pruned), log, False)

            if args.use_cuda:
                torch.cuda.empty_cache()
                model_pruned.cuda()

            pruned_size = get_n_params_(model_pruned)
            pruned_ops = get_n_flops_(model_pruned, img_size=(224, 224))
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

def prune(model, train_loader, train_sampler, loader_subset, test_loader, criterion, log):

    sub_inputs, sub_targets = get_train_subset_in_memory(loader_subset)

    with torch.no_grad():
        val_acc_top1, val_acc_top5, val_loss = validate(test_loader, model, criterion)
        print_log("Val Acc@1: %0.3lf, Acc@5: %0.3lf,  Loss: %0.5f" % (val_acc_top1, val_acc_top5, val_loss), log) 

    flops_baseline = get_n_flops_(model, img_size=(224, 224))

    current_flop_reduction_rate = 0.0
    flop_reduction_rate_temp = 0.0
    filter_prune_limit_per_layer = 0.75

    model.eval()

    layers_step_prune_count = calc_conv_layers_step_prune_count(model)
    print("layers_step_prune_count", layers_step_prune_count)

    layerwise_filter_count_org = get_conv_filter_count_v1(model)
    print("layerwise_filter_count_org", layerwise_filter_count_org)

    print("Start Pruning ...")    

    methods = ['l1norm', 'l2norm', 'eucl', 'cos']
    prune_stat = {}
    for m in methods:
        prune_stat[m] = 0

    while current_flop_reduction_rate < args.rate_flop:
        small_loss = 100000000000.0
        small_loss_lindex = 0
        opt_method = ''

        for prune_conv_idx in layerwise_filter_count_org.keys():
            for method in methods:
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

        flops_pruned = get_n_flops_(model, img_size=(224, 224))
        current_flop_reduction_rate = 1.0 - flops_pruned / flops_baseline
        print_log("[Pruning Method: %s] Flop Reduction Rate: %lf/%lf [Pruned %d filters from %s]" % (opt_method,
                    current_flop_reduction_rate, args.rate_flop, layers_step_prune_count[small_loss_lindex], small_loss_lindex), log)

        if current_flop_reduction_rate - flop_reduction_rate_temp > 0.03:
            if args.use_cuda:
                torch.cuda.empty_cache()
                model.cuda()
            # train 
            model = train(model, train_loader, test_loader, train_sampler, criterion, args.prune_epoch, args.prune_epoch  + args.recover_epoch, log)

            flop_reduction_rate_temp = current_flop_reduction_rate

    print_log('Prune Stats: ' + ''.join(str(prune_stat)), log)

    layerwise_filter_count_prune = get_conv_filter_count_v1(model)
    print_log('Final Flop Reduction Rate: %.4lf' % current_flop_reduction_rate, log)
    print_log("Conv Filters Before Pruning: " + ''.join(str(layerwise_filter_count_org)), log)
    print_log("Conv Filters After Pruning: " + ''.join(str(layerwise_filter_count_prune)), log)    

    filter_prune_rate = {}
    for idx in layerwise_filter_count_org.keys():
        filter_prune_rate[idx] = 1.0 - float(layerwise_filter_count_prune[idx]/float(layerwise_filter_count_org[idx]))
    print_log("Layerwise Pruning Rate: " + ''.join(str(filter_prune_rate)), log)    

    # save pruned model before finetuning
    save_checkpoint({
        'epoch': 0,
        'arch': args.arch,
        'state_dict': model,
    }, False, args.save_path, '%s.init.prune.pth.tar' % (args.arch) )

    return model

# this function will return number of filters to be pruned from each layer 
# that results in ~1% reduction in FLOPs
def calc_conv_layers_step_prune_count(model):
    conv_filter_count = []
    for idx, m in enumerate(model.modules()):
        if isinstance(m, nn.Conv2d):
            conv_filter_count.append(idx)

    layers_step_prune_count = {}
    for id in conv_filter_count:  
        model_org = copy.deepcopy(model)
        input = torch.randn(1,3,224,224)
        DG = tp.DependencyGraph().build_dependency(model_org, input.cuda())
        flops_baseline = get_n_flops_(model_org, img_size=(224, 224))  
        for idx, m in enumerate(model_org.modules()):
            if isinstance(m, nn.Conv2d) and idx == id:
                pruning_idxs = norm_prune(m, 1)
                plan = DG.get_pruning_plan(m, tp.prune_conv_out_channel, pruning_idxs)
                plan.exec() 
                break
        flops_pruned = get_n_flops_(model_org, img_size=(224, 224))
        filter_prune_step = int(1.0 / ((1.0 - flops_pruned / flops_baseline) * 100) + 0.5)
        layers_step_prune_count[idx] = filter_prune_step
    
    return layers_step_prune_count

def get_conv_filter_count_v1(model):
    conv_filter_count = {}    
    for idx, m in enumerate(model.modules()):
        if isinstance(m, nn.Conv2d):
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

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def init_distributed_mode(args):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
    elif hasattr(args, "rank"):
        pass
    else:
        print("Not using distributed mode")
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl"
    print(f"| distributed init (rank {args.rank}): {args.dist_url}", flush=True)
    torch.distributed.init_process_group(
        backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank
    )
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)

def train(model, train_loader, test_loader, train_sampler, criterion, start_epoch, total_epoches, log, optimizer=None):

    if optimizer is None:
        optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum,
            weight_decay=args.decay, nesterov=True)

    if args.use_cuda:
        torch.cuda.empty_cache()
        model.cuda()

    recorder = RecorderMeter(args.total_epoches)
    start_time = time.time()
    epoch_time = AverageMeter()
    best_accuracy = 0

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    # Main loop
    for epoch in range(start_epoch, total_epoches):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        current_learning_rate = adjust_learning_rate(optimizer, epoch)
        #current_learning_rate = adjust_learning_rate_vgg(optimizer, epoch)

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
            'state_dict': model_without_ddp,
            'recorder': recorder,
            'optimizer': optimizer.state_dict(),
        }, is_best, args.save_path, '%s.checkpoint.pth.tar' % (args.arch))

        # measure elapsed time
        epoch_time.update(time.time() - start_time)
        start_time = time.time()
        recorder.plot_curve(os.path.join(args.save_path, 'curve.png'))
        print_log('Epoch: [{0}]\tTime {epoch_time.val:.3f} ({epoch_time.avg:.3f})'.format(epoch, epoch_time=epoch_time), log)

    return model_without_ddp

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

        if i % args.print_freq == 0:
            print_log('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5), log)

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

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every args.decay_epoch_step epochs"""
    lr = args.lr * (0.1 ** (epoch // args.decay_epoch_step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

if __name__=='__main__':
    main()