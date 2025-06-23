import argparse
import datetime
import math
import os
import time
import torch
import torch.distributed.optim
import torch.nn.functional as F
import torch.utils.data
import torchvision
from spikingjelly.activation_based import functional, surrogate
from torch import nn
from torch.cuda import amp
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

import sew_resnet
import utils

_seed_ = 41
import random
random.seed(_seed_)

torch.manual_seed(_seed_)  # use torch.manual_seed() to seed the RNG for all devices (both CPU and CUDA)
torch.cuda.manual_seed_all(_seed_)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import numpy as np
np.random.seed(_seed_)


class SurrogateSampling(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return torch.bernoulli(input)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class SurrogateSigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, alpha):
        ctx.save_for_backward(input)
        ctx.alpha = alpha
        sgax = (input * ctx.alpha).sigmoid_()
        return sgax

    @staticmethod
    def backward(ctx, grad_output):
        alpha_back = torch.max(torch.tensor([1, ctx.alpha / 3]))
        sgax = (ctx.saved_tensors[0] * alpha_back).sigmoid_()
        tmp = grad_output * (1.0 - sgax) * sgax
        grad_x = tmp * alpha_back
        return grad_x, None

delta = 0.5

class ZO(torch.autograd.Function):  # https://github.com/BhaskarMukhoty/LocalZO
    @staticmethod
    def forward(ctx, input, delta=delta):
        out = (input > 0).float()
        L = torch.tensor([delta])
        ctx.save_for_backward(input, L)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (input, others) = ctx.saved_tensors
        delta = others[0].item()
        grad_input = grad_output.clone()

        sample_size = 5
        abs_z = torch.abs(torch.randn((sample_size,) + input.size(), device=torch.device('cuda'), dtype=torch.float))
        t = torch.abs(input[None, :, :]) < abs_z * delta
        grad_input = grad_input * torch.mean(t * abs_z, dim=0) / (2 * delta)

        return grad_input, None

surr_ZO = ZO.apply

class SurrogateSigmoid_learnable_alpha(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, alpha):
        ctx.save_for_backward(input)
        ctx.alpha = alpha
        sgax = (input * ctx.alpha).sigmoid_()
        return sgax

    @staticmethod
    def backward(ctx, grad_output):
        sgax = (ctx.saved_tensors[0] * ctx.alpha / 3).sigmoid_()
        tmp = grad_output * (1.0 - sgax) * sgax
        grad_x = tmp * ctx.alpha / 3
        grad_alpha = tmp * ctx.saved_tensors[0]
        return grad_x, grad_alpha

class IFNode5(nn.Module): # https://github.com/fangwei123456/Parallel-Spiking-Neuron
    def __init__(self, T: int, surrogate_function: surrogate.SurrogateFunctionBase):
        super().__init__()
        self.surrogate_function = surrogate_function
        self.fc = nn.Linear(T, T)
        nn.init.constant_(self.fc.bias, -1)

    def forward(self, x_seq: torch.Tensor):
        # x_seq.shape = [T, N, *]
        h_seq = torch.addmm(self.fc.bias.unsqueeze(1), self.fc.weight, x_seq.flatten(1))
        spike = self.surrogate_function(h_seq)
        return spike.view(x_seq.shape)



class LIF(nn.Module):
    def __init__(self, T: int, surrogate_function: surrogate.SurrogateFunctionBase, device=None):
        super().__init__()
        self.T = T
        # self.surrogate_function = StochasticST.apply
        self.surrogate_function = surr_ZO
        self.threshold = torch.tensor([1.0], device=device)
        self.maxiter = torch.tensor([4], device=device)
        self.lam = torch.tensor([0.5], device=device)
        # self.tau = nn.Parameter(torch.tensor([0.], device=device))
        self.device = device

    def forward(self, x_seq: torch.Tensor):
        # x_seq.shape = [N, T, *]

        # 调整 x_seq 形状以适应矩阵乘法
        batch_size, T = x_seq.shape[:2]
        remaining_dims = x_seq.shape[2:]

        o = torch.zeros_like(x_seq)
        u = torch.zeros_like(x_seq[:, 0, ...])
        # print(self.lam.device)
        for t in range(T):
            u = self.lam * u + x_seq[:, t, ...]
            o[:, t, ...] = self.surrogate_function(u - self.threshold)
            u = u - o[:, t, ...]
        return o


class FPT_learnable_alpha(nn.Module):
    def __init__(self, T: int, surrogate_function: surrogate.SurrogateFunctionBase, device=None):
        super().__init__()
        self.T = T
        self.SurrogateSampling = SurrogateSampling.apply
        self.SurrogateSigmoid = SurrogateSigmoid_learnable_alpha.apply
        self.threshold = torch.tensor([1.0], device=device)
        self.max_iter = torch.tensor([2], device=device)
        self.lam = torch.tensor([0.5], device=device)
        self.device = device
        self.alpha = nn.Parameter(torch.tensor([4.], device=device))

        self.Lambda = self._generate_lambda_matrix()
        self.A = self.Lambda - torch.eye(self.T, device=self.device)

    def _generate_lambda_matrix(self):
        Lambda = torch.zeros((self.T, self.T), device=self.device)
        for i in range(self.T):
            for j in range(i + 1):
                Lambda[i, j] = self.lam ** (i - j)
        return Lambda

    def forward(self, x_seq: torch.Tensor):
        # x_seq.shape = [T, N, *]
        T, batch_size = x_seq.shape[:2]
        remaining_dims = x_seq.shape[2:]
        # x_reshaped = x_seq.view(T, batch_size, -1).transpose(0, 1)

        u = torch.matmul(self.Lambda, x_seq.flatten(1))
        o = self.SurrogateSigmoid(u - self.threshold, self.alpha)
        C = u

        for _ in range(self.max_iter):
            u = -self.threshold * torch.matmul(self.A, o) + C
            o = self.SurrogateSigmoid(u - self.threshold, self.alpha)

        o = o.view(T, batch_size, *remaining_dims)
        o = self.SurrogateSampling(o)
        return o

class FPT_adaptive_alpha(nn.Module):
    def __init__(self, T: int, surrogate_function: surrogate.SurrogateFunctionBase, device=None):
        super().__init__()
        self.T = T
        self.SurrogateSampling = SurrogateSampling.apply
        self.SurrogateSigmoid = SurrogateSigmoid.apply
        self.threshold = torch.tensor([1.0], device=device)
        self.max_iter = torch.tensor([2], device=device)
        self.lam = torch.tensor([0.5], device=device)
        self.device = device
        self.alpha = torch.tensor([2.], device=device)

        self.Lambda = self._generate_lambda_matrix()
        self.A = self.Lambda - torch.eye(self.T, device=self.device)

    def _generate_lambda_matrix(self):
        Lambda = torch.zeros((self.T, self.T), device=self.device)
        for i in range(self.T):
            for j in range(i + 1):
                Lambda[i, j] = self.lam ** (i - j)
        return Lambda

    def forward(self, x_seq: torch.Tensor):
        T, batch_size = x_seq.shape[:2]
        remaining_dims = x_seq.shape[2:]


        u = torch.matmul(self.Lambda, x_seq.flatten(1))
        o = self.SurrogateSigmoid(u - self.threshold, self.alpha)
        C = u

        # for k in range(self.max_iter):
        #     u = -self.threshold * torch.matmul(self.A, o) + C
        #     o = self.SurrogateSigmoid(u - self.threshold, self.alpha * (3 ** (1 + k)))

        u = -self.threshold * torch.matmul(self.A, o) + C
        o = self.SurrogateSigmoid(u - self.threshold, self.alpha * 3)
        u = -self.threshold * torch.matmul(self.A, o) + C
        o = self.SurrogateSigmoid(u - self.threshold, self.alpha * 6)

        o = o.view(T, batch_size, *remaining_dims)
        o = self.SurrogateSampling(o)
        return o

def ce_loss(y, target):
    return F.cross_entropy(y.mean(0), target)

def TET_loss(outputs, labels, lamb):
    Loss_es = ce_loss(outputs, labels)
    if lamb != 0:
        Loss_mmd = functional.temporal_efficient_training_cross_entropy(outputs, labels)
    else:
        Loss_mmd = 0
    return (1-lamb) * Loss_es + lamb * Loss_mmd # L_Total

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, lamb=0, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    metric_logger.add_meter('img/s', utils.SmoothedValue(window_size=10, fmt='{value}'))

    header = 'Epoch: [{}]'.format(epoch)

    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        start_time = time.time()
        image, target = image.to(device), target.to(device)
        # with torch.autograd.detect_anomaly():
        if scaler is not None:
            with amp.autocast():
                output = model(image)
                loss = TET_loss(output, target, lamb)
        else:
            output = model(image)
            loss = TET_loss(output, target, lamb)

        optimizer.zero_grad()

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        else:
            loss.backward()
            optimizer.step()

        functional.reset_net(model)
        if output.dim() == 3:
            with torch.no_grad():
                output = output.mean(0)

        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        batch_size = image.shape[0]
        loss_s = loss.item()
        if math.isnan(loss_s):
            raise ValueError('loss is Nan')
        acc1_s = acc1.item()
        acc5_s = acc5.item()

        metric_logger.update(loss=loss_s, lr=optimizer.param_groups[0]["lr"])

        # end_time = time.time()
        # print(f"Time taken for one batch: {end_time - start_time} seconds")

        metric_logger.meters['acc1'].update(acc1_s, n=batch_size)
        metric_logger.meters['acc5'].update(acc5_s, n=batch_size)
        metric_logger.meters['img/s'].update(batch_size / (time.time() - start_time))

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    return metric_logger.loss.global_avg, metric_logger.acc1.global_avg, metric_logger.acc5.global_avg



def evaluate(model, data_loader, device, print_freq=100, lamb=0, header='Test:'):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, print_freq, header):
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(image)
            loss = TET_loss(output, target, lamb)
            functional.reset_net(model)
            if output.dim() == 3:
                with torch.no_grad():
                    output = output.mean(0)

            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            # FIXME need to take into account that the datasets
            # could have been padded in distributed setup
            batch_size = image.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    loss, acc1, acc5 = metric_logger.loss.global_avg, metric_logger.acc1.global_avg, metric_logger.acc5.global_avg
    print(f' * Acc@1 = {acc1}, Acc@5 = {acc5}, loss = {loss}')
    return loss, acc1, acc5


def _get_cache_path(filepath):
    import hashlib
    h = hashlib.sha1(filepath.encode()).hexdigest()
    cache_path = os.path.join("~", ".torch", "vision", "datasets", "imagefolder", h[:10] + ".pt")
    cache_path = os.path.expanduser(cache_path)
    return cache_path

def load_data(traindir, valdir, cache_dataset, distributed):
    # Data loading code
    print("Loading data")
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    print("Loading training data")
    st = time.time()
    cache_path = _get_cache_path(traindir)
    if cache_dataset and os.path.exists(cache_path):
        # Attention, as the transforms are also cached!
        print("Loading dataset_train from {}".format(cache_path))
        dataset, _ = torch.load(cache_path)
    else:
        dataset = torchvision.datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),  # 随机颜色抖动
                # transforms.RandomRotation(15),  # 随机旋转
                transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.IMAGENET),
                transforms.ToTensor(),
                normalize,
            ]))
        if cache_dataset:
            print("Saving dataset_train to {}".format(cache_path))
            utils.mkdir(os.path.dirname(cache_path))
            utils.save_on_master((dataset, traindir), cache_path)
    print("Took", time.time() - st)

    print("Loading validation data")
    cache_path = _get_cache_path(valdir)
    if cache_dataset and os.path.exists(cache_path):
        # Attention, as the transforms are also cached!
        print("Loading dataset_test from {}".format(cache_path))
        dataset_test, _ = torch.load(cache_path)
    else:
        dataset_test = torchvision.datasets.ImageFolder(
            valdir,
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]))
        if cache_dataset:
            print("Saving dataset_test to {}".format(cache_path))
            utils.mkdir(os.path.dirname(cache_path))
            utils.save_on_master((dataset_test, valdir), cache_path)

    print("Creating data loaders")
    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    return dataset, dataset_test, train_sampler, test_sampler


def main(args):


    max_test_acc1 = 0.
    test_acc5_at_max_test_acc1 = 0.


    train_tb_writer = None
    te_tb_writer = None


    utils.init_distributed_mode(args)
    print(args)
    output_dir = os.path.join(args.output_dir, f'{args.model}_b{args.batch_size}_lr{args.lr}_T{args.T}')
    output_model_dir = os.path.join('./pt_imagenet100', f'{args.model}_b{args.batch_size}_lr{args.lr}_T{args.T}')

    # timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    # output_dir += f'_{timestamp}'

    if args.weight_decay:
        output_dir += f'_wd{args.weight_decay}'
        output_model_dir += f'_wd{args.weight_decay}'

    if args.cos_lr_T == -1:
        args.cos_lr_T = args.epochs

    output_dir += f'_coslr{args.cos_lr_T}'
    output_model_dir += f'_coslr{args.cos_lr_T}'

    output_dir += f'_{args.opt}'
    output_model_dir += f'_{args.opt}'

    output_dir += f'_{args.world_size}gpu'
    output_model_dir += f'_{args.world_size}gpu'

    output_dir += f'_lamb{args.lamb}'
    output_model_dir += f'_lamb{args.lamb}'

    output_dir += f'_{args.neu}'
    output_model_dir += f'_{args.neu}'

    if args.load is not None:
        output_dir += '_load'
        output_model_dir += '_load'

    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    output_dir += f'_{timestamp}'
    output_dir += '_lily'
    output_model_dir += f'_{timestamp}'
    output_model_dir += '_lily'


    if output_dir:
        # utils.mkdir(output_dir)
        utils.mkdir(output_model_dir)

    print(output_dir)
    print(output_model_dir)


    device = torch.device(args.device)

    train_dir = os.path.join(args.data_path, 'train')
    val_dir = os.path.join(args.data_path, 'val')
    dataset_train, dataset_test, train_sampler, test_sampler = load_data(train_dir, val_dir,
                                                                   args.cache_dataset, args.distributed)
    print(f'dataset_train:{dataset_train.__len__()}, dataset_test:{dataset_test.__len__()}')

    data_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=args.batch_size,
        sampler=train_sampler, num_workers=args.workers, pin_memory=True)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size,
        sampler=test_sampler, num_workers=args.workers, pin_memory=True)

    print("Creating model")

    if args.model in sew_resnet.__dict__:
        if args.neu == 'FPT_learnable_alpha':
            model = sew_resnet.__dict__[args.model](pretrained=False, cnf='ADD',
                                                    spiking_neuron=FPT_learnable_alpha,
                                                    surrogate_function=surrogate.ATan(), T=args.T, device=args.device)
        elif args.neu == 'FPT_adaptive_alpha':
            model = sew_resnet.__dict__[args.model](pretrained=False, cnf='ADD',
                                                    spiking_neuron=FPT_adaptive_alpha,
                                                    surrogate_function=surrogate.ATan(), T=args.T, device=args.device)
        elif args.neu == 'LIF':
            model = sew_resnet.__dict__[args.model](pretrained=False, cnf='ADD', spiking_neuron=LIF, surrogate_function=surrogate.ATan(), T=args.T, device=args.device)
    else:
        raise NotImplementedError(args.model)


    print(model)

    if args.load is not None:
        model.load_state_dict(torch.load(args.load), strict=False)
        print('load', args.load)


    model.to(device)
    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)


    if args.opt == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.opt == 'adam':
         optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))
    else:
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    if args.amp:
        scaler = amp.GradScaler()
    else:
        scaler = None

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.cos_lr_T)



    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

        args.start_epoch = checkpoint['epoch'] + 1

        max_test_acc1 = checkpoint['max_test_acc1']
        test_acc5_at_max_test_acc1 = checkpoint['test_acc5_at_max_test_acc1']

    if args.test_only:
        evaluate(model, data_loader_test, device=device, lamb=args.lamb, header='Test:')
        return

    if args.tb and utils.is_main_process():
        purge_step_train = args.start_epoch
        purge_step_te = args.start_epoch
        train_tb_writer = SummaryWriter(output_dir + '_logs/train', purge_step=purge_step_train)
        te_tb_writer = SummaryWriter(output_dir + '_logs/te', purge_step=purge_step_te)
        with open(output_dir + '_logs/args.txt', 'w', encoding='utf-8') as args_txt:
            args_txt.write(str(args))

        print(f'purge_step_train={purge_step_train}, purge_step_te={purge_step_te}')

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        save_max = False
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train_loss, train_acc1, train_acc5 = train_one_epoch(model, optimizer, data_loader, device, epoch,  args.print_freq, args.lamb,  scaler)
        if utils.is_main_process():
            train_tb_writer.add_scalar('train_loss', train_loss, epoch)
            train_tb_writer.add_scalar('train_acc1', train_acc1, epoch)
            train_tb_writer.add_scalar('train_acc5', train_acc5, epoch)
        lr_scheduler.step()

        test_loss, test_acc1, test_acc5 = evaluate(model, data_loader_test, device=device, lamb=args.lamb, header='Test:')
        if te_tb_writer is not None:
            if utils.is_main_process():

                te_tb_writer.add_scalar('test_loss', test_loss, epoch)
                te_tb_writer.add_scalar('test_acc1', test_acc1, epoch)
                te_tb_writer.add_scalar('test_acc5', test_acc5, epoch)


        if max_test_acc1 < test_acc1:
            max_test_acc1 = test_acc1
            test_acc5_at_max_test_acc1 = test_acc5
            save_max = True



        if output_model_dir:

            checkpoint = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args,
                'max_test_acc1': max_test_acc1,
                'test_acc5_at_max_test_acc1': test_acc5_at_max_test_acc1,
            }

            utils.save_on_master(
                checkpoint,
                os.path.join(output_model_dir, 'checkpoint_latest.pth'))
            save_flag = False

            if epoch % 64 == 0 or epoch == args.epochs - 1:
                save_flag = True

            elif args.cos_lr_T == 0:
                for item in args.lr_step_size:
                    if (epoch + 2) % item == 0:
                        save_flag = True
                        break

            if save_flag:
                utils.save_on_master(
                    checkpoint,
                    os.path.join(output_model_dir, f'checkpoint_{epoch}.pth'))

            if save_max:
                utils.save_on_master(
                    checkpoint,
                    os.path.join(output_model_dir, 'checkpoint_max_test_acc1.pth'))
        print(args)
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(output_model_dir)

        print('Training time {}'.format(total_time_str), 'max_test_acc1', max_test_acc1,
              'test_acc5_at_max_test_acc1', test_acc5_at_max_test_acc1)

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Classification Training')

    parser.add_argument('--data-path', default='/data', help='dataset')

    parser.add_argument('--model', default='sew_resnet34', help='model')
    parser.add_argument('--device', default='cuda:3', help='device')
    parser.add_argument('-b', '--batch-size', default=64, type=int)
    parser.add_argument('--epochs', default=320, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument('--lr', default=0.005, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='Momentum for SGD. Adam will not use momentum')
    parser.add_argument('--wd', '--weight-decay', default=5e-5, type=float,
                        metavar='W', help='weight decay (default: 0)', dest='weight_decay')
    parser.add_argument('--opt', default='sgd', type=str, help='optimizer')
    parser.add_argument('--print-freq', default=1016, type=int, help='print frequency')
    parser.add_argument('--output-dir', default='./logs_imagenet100', help='path where to save')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--lamb',
                        default=1,
                        type=float,
                        metavar='N',
                        help='tet loss  lambda')
    parser.add_argument('--neu', default='FPT_adaptive_alpha', type=str)
    parser.add_argument(
        "--cache-dataset",
        dest="cache_dataset",
        help="Cache the datasets for quicker initialization. It also serializes the transforms",
        default=False,
    )
    parser.add_argument(
        "--sync-bn",
        dest="sync_bn",
        help="Use sync batch norm",
        action="store_true",
    )
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )

    # Mixed precision training parameters
    parser.add_argument('--amp', default=True,
                        help='Use AMP training')


    # distributed training parameters
    parser.add_argument('--world-size', default=8, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')


    parser.add_argument('--tb', default=True,
                        help='Use TensorBoard to record logs')
    parser.add_argument('--T', default=4, type=int, help='simulation steps')

    parser.add_argument('--cos_lr_T', default=320, type=int,
                        help='T_max of CosineAnnealingLR.')


    parser.add_argument('--load', type=str, default='./resnet34.pth', help='the pt file path for loading pre-trained ANN weights')


    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)


