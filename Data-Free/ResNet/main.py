'''
    This code is based on the official PyTorch ImageNet training example 'main.py'. Commit ID: fa9584456cced1d3a6cf19869ca6fb912ecee778, March 17th, 2022.
    URL: https://github.com/pytorch/examples/tree/main/imagenet
'''

import argparse
from hashlib import new
import os
import random
import shutil
import time
import warnings
import copy
from enum import Enum
from collections import OrderedDict
from pruner.genthin import GenThinPruner
from tqdm import tqdm
from torchvision.datasets import CIFAR10
from ptflops import get_model_complexity_info

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
from torch.optim.lr_scheduler import StepLR, MultiStepLR
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='DFPC PyTorch Implementation')
parser.add_argument('--data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--dataset',
                    help='dataset name', choices=['imagenet', 'cifar10', 'cifar100'])
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='learning rate decay factor', dest='gamma')
parser.add_argument('--decay-step-size', default=5, type=int,
                    help='number of epochs after which learning rate is reduced', dest='decay_step_size')
parser.add_argument('--accuracy-threshold', default=2.5, type=float,
                    help='validation accuracy drop feasible for the pruned model', dest='accuracy_threshold')
parser.add_argument('--accuracy-drop-to-stop', default=1, type=float,
                    help='stopping criterion. validation accuracy drop, beyond feasible accuracy, when we stop pruning', \
                    dest='accuracy_drop_to_stop')
parser.add_argument('--pruning-percentage', default=0.01, type=float,
                    help='percentage of channels to prune per pruning iteration', dest='pruning_percentage')
parser.add_argument('--num-processes', default=5, type=int, # More the merrier, but RAM consumption will increase drastically.
                    help='number of simultaneous process to spawn for multiprocessing', dest='num_processors')
parser.add_argument('--scoring-strategy', default='dfpc', type=str,
                    help='strategy to compute saliencies of channels', dest='strategy',
                    choices=['dfpc', 'l1', 'random'])
parser.add_argument('--prune-coupled', default=1, type=int,
                    help='prune coupled channels is set to 1', dest='prunecoupled',
                    choices=[0, 1])

args = parser.parse_args()

best_acc1 = 0
if args.dataset in ['cifar10']:
    from models import *
elif args.dataset in ['cifar100']:
    from models_100 import *

def main():
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    main_worker(args.gpu, args)


def main_worker(gpu, args):
    global best_acc1
    args.gpu = gpu

    # Data loading code
    if args.dataset in ['cifar100']:
        data_path = './data'

        transform_train = transforms.Compose([
                            #transforms.ToPILImage(),
                            transforms.RandomCrop(32, padding=4),
                            transforms.RandomHorizontalFlip(),
                            transforms.RandomRotation(15),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), #mean
                                                    (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)) #stddev
                        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), #mean
                                    (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)) #stddev
        ])

        train_set = torchvision.datasets.CIFAR100(
                        root='./data', train=True, download=True, transform=transform_train)
        test_set = torchvision.datasets.CIFAR100(
                        root='./data', train=False, download=True, transform=transform_test)

        train_loader = torch.utils.data.DataLoader(
                        train_set, batch_size=128, shuffle=True, num_workers=2)
        val_loader = torch.utils.data.DataLoader(
                            test_set, batch_size=100, shuffle=False, num_workers=2)
    elif args.dataset in ['cifar10']:
        data_path = './data'

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]) # these mean and var are from official PyTorch ImageNet example
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
        ])

        train_set = CIFAR10(data_path,
                            train=True,
                            download=True,
                            transform=transform_train)
        test_set = CIFAR10(data_path,
                            train=False,
                            download=True,
                            transform=transform_test)

        train_loader = torch.utils.data.DataLoader(train_set,
                                batch_size=args.batch_size, #keep it 128
                                num_workers=args.workers,
                                shuffle=True,
                                pin_memory=True)
        val_loader = torch.utils.data.DataLoader(test_set,
                                    batch_size=args.batch_size,
                                    num_workers=args.workers,
                                    shuffle=False,
                                    pin_memory=True)
    else:
        traindir = os.path.join(args.data, 'train')
        valdir = os.path.join(args.data, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])

        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))

        train_sampler = None

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=train_sampler)

        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
    
    # loading model for pruning

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint")
            base_model, unpruned_accuracy, pruning_iteration = LoadBaseModel()
            print("=> loaded checkpoint")
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        pruning_iteration = 0
        # create model
        if args.dataset in ['cifar100']:
            if 'resnet50' in args.arch:
                print("=> using pre-trained model of 'resnet50'")
                base_model = ResNet50()
                state_dict = torch.load('./pretrained_checkpoints/cifar100_resnet50_ckpt.pth')['net']
            elif 'resnet101' in args.arch:
                print("=> using pre-trained model of 'resnet101'")
                base_model = ResNet101()
                state_dict = torch.load('./pretrained_checkpoints/cifar100_resnet101_ckpt.pth')['net']
            base_model.load_state_dict(DataParallelStateDict_To_StateDict(state_dict))
            del state_dict
        elif args.dataset in ['cifar10']:
            if 'resnet50' in args.arch:
                print("=> using pre-trained model of 'resnet50'")
                base_model = ResNet50()
                state_dict = torch.load('./pretrained_checkpoints/cifar10_resnet50_ckpt.pth')['net']
            elif 'resnet101' in args.arch:
                print("=> using pre-trained model of 'resnet101'")
                base_model = ResNet101()
                state_dict = torch.load('./pretrained_checkpoints/cifar10_resnet101_ckpt.pth')['net']
            base_model.load_state_dict(DataParallelStateDict_To_StateDict(state_dict))
            del state_dict
        elif 'resnet' in args.arch:
            print("=> using pre-trained model of '{}'".format(args.arch))
            base_model = models.__dict__[args.arch](pretrained=True).to('cpu')
        else:
            raise ValueError('Specified model architecture is not supported.')
    net = model = copy.deepcopy(base_model)
    macs, params = get_model_complexity_info(net, (3, 32, 32), as_strings=True, print_per_layer_stat=False)
    del net
    model = ToAppropriateDevice(copy.deepcopy(base_model), args)

    cudnn.benchmark = True

    # define loss function (criterion)
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    accuracy = validate(val_loader, model, criterion, args)
    print('-{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('+{:<30}  {:<8}'.format('Number of parameters: ', params))
    
    if args.evaluate:
        return
    
    if args.resume:
        pass
    else:
        unpruned_accuracy = accuracy

    print('Initializing Pruner...')
    pruner = GenThinPruner(base_model, args)
    print('Computing Saliency Scores...')
    pruner.ComputeSaliencyScores(base_model)
    while accuracy >= 10.5:
        pruning_iteration += 1
        print('Pruning iteration {}...'.format(pruning_iteration))
        print('Pruning the model...')
        num_channels_to_prune = int(args.pruning_percentage * pruner.total_channels(base_model))
        for _ in range(num_channels_to_prune):
            pruner.Prune(base_model)

        model = copy.deepcopy(base_model)
        model = torch.nn.DataParallel(model).cuda()

        acc1 = validate(val_loader, model, criterion, args)

        # remember best pruned model and save checkpoint
        is_best = (acc1 >= unpruned_accuracy - args.accuracy_threshold)

        save_checkpoint({
            'pruning_iteration': pruning_iteration,
            'unpruned_accuracy': unpruned_accuracy,
            'arch': args.arch,
            'model': model,
            'state_dict': model.state_dict(),
            'acc1': acc1,
        }, is_best, filename='dataparallel_model.pth.tar')

        save_checkpoint({
                    'arch': args.arch,
                    'model': base_model
                }, is_best, filename='base_model.pth.tar')
    
        del model, base_model

        base_model, _, _ = LoadBaseModel()
        net = model = copy.deepcopy(base_model)
        macs, params = get_model_complexity_info(net, (3, 32, 32), as_strings=True, print_per_layer_stat=False)
        del net
        print('-{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('+{:<30}  {:<8}'.format('Number of parameters: ', params))
        accuracy = acc1

def finetune(model, train_loader, val_loader, criterion, optimizer, scheduler):
    global best_acc1
    for epoch in range(args.start_epoch, args.epochs):
        print('Finetuning epoch {} of {} with learning rate {}.'.format(epoch+1, args.epochs, scheduler.get_last_lr()))

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, args)
        
        scheduler.step()
    return acc1


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(tqdm(train_loader)):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        if torch.cuda.is_available():
            target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(tqdm(val_loader)):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        progress.display_summary()

    return top1.avg.item()


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'best_'+filename)

class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)
        
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries), flush = True)
        
    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries), flush = True)

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def DataParallelStateDict_To_StateDict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    return new_state_dict

def ToAppropriateDevice(model, args):
    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()
    return model

def LoadBaseModel():
    base_model_dict = torch.load('base_model.pth.tar', map_location=torch.device('cpu'))
    base_model = base_model_dict['model']
    model_dict = torch.load('dataparallel_model.pth.tar', map_location=torch.device('cpu'))
    state_dict = model_dict['state_dict']
    unpruned_accuracy = model_dict['unpruned_accuracy']
    pruning_iteration = model_dict['pruning_iteration']
    base_model.load_state_dict(DataParallelStateDict_To_StateDict(state_dict))
    del base_model_dict, model_dict, state_dict
    base_model = base_model.to('cpu')
    return base_model, unpruned_accuracy, pruning_iteration

if __name__ == '__main__':
    main()
