import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import random

import torchvision
from torchvision import transforms
from spikingjelly.activation_based import neuron, functional, surrogate, layer
from torch.utils.tensorboard import SummaryWriter
import os
import time
import argparse
from torch.cuda import amp
import sys
import datetime
from torch.utils.data.dataloader import default_collate
from torchvision.transforms import autoaugment, transforms
from torchvision.transforms.functional import InterpolationMode
from torch import Tensor
from typing import Tuple

class StochasticST(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return torch.bernoulli(input)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

def atan(x: torch.Tensor, alpha: float):
    return (math.pi / 2 * alpha * x).atan_() / math.pi + 0.5

class RandomMixup(torch.nn.Module):
    """Randomly apply Mixup to the provided batch and targets.
    The class implements the data augmentations as described in the paper
    `"mixup: Beyond Empirical Risk Minimization" <https://arxiv.org/abs/1710.09412>`_.

    Args:
        num_classes (int): number of classes used for one-hot encoding.
        p (float): probability of the batch being transformed. Default value is 0.5.
        alpha (float): hyperparameter of the Beta distribution used for mixup.
            Default value is 1.0.
        inplace (bool): boolean to make this transform inplace. Default set to False.
    """

    def __init__(self, num_classes: int, p: float = 0.5, alpha: float = 1.0, inplace: bool = False) -> None:
        super().__init__()
        assert num_classes > 0, "Please provide a valid positive value for the num_classes."
        assert alpha > 0, "Alpha param can't be zero."

        self.num_classes = num_classes
        self.p = p
        self.alpha = alpha
        self.inplace = inplace

    def forward(self, batch: Tensor, target: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            batch (Tensor): Float tensor of size (B, C, H, W)
            target (Tensor): Integer tensor of size (B, )

        Returns:
            Tensor: Randomly transformed batch.
        """
        if batch.ndim != 4:
            raise ValueError(f"Batch ndim should be 4. Got {batch.ndim}")
        if target.ndim != 1:
            raise ValueError(f"Target ndim should be 1. Got {target.ndim}")
        if not batch.is_floating_point():
            raise TypeError(f"Batch dtype should be a float tensor. Got {batch.dtype}.")
        if target.dtype != torch.int64:
            raise TypeError(f"Target dtype should be torch.int64. Got {target.dtype}")

        if not self.inplace:
            batch = batch.clone()
            target = target.clone()

        if target.ndim == 1:
            target = torch.nn.functional.one_hot(target, num_classes=self.num_classes).to(dtype=batch.dtype)

        if torch.rand(1).item() >= self.p:
            return batch, target

        # It's faster to roll the batch by one instead of shuffling it to create image pairs
        batch_rolled = batch.roll(1, 0)
        target_rolled = target.roll(1, 0)

        # Implemented as on mixup paper, page 3.
        lambda_param = float(torch._sample_dirichlet(torch.tensor([self.alpha, self.alpha]))[0])
        batch_rolled.mul_(1.0 - lambda_param)
        batch.mul_(lambda_param).add_(batch_rolled)

        target_rolled.mul_(1.0 - lambda_param)
        target.mul_(lambda_param).add_(target_rolled)

        return batch, target

    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}("
            f"num_classes={self.num_classes}"
            f", p={self.p}"
            f", alpha={self.alpha}"
            f", inplace={self.inplace}"
            f")"
        )
        return s


class RandomCutmix(torch.nn.Module):
    """Randomly apply Cutmix to the provided batch and targets.
    The class implements the data augmentations as described in the paper
    `"CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features"
    <https://arxiv.org/abs/1905.04899>`_.

    Args:
        num_classes (int): number of classes used for one-hot encoding.
        p (float): probability of the batch being transformed. Default value is 0.5.
        alpha (float): hyperparameter of the Beta distribution used for cutmix.
            Default value is 1.0.
        inplace (bool): boolean to make this transform inplace. Default set to False.
    """

    def __init__(self, num_classes: int, p: float = 0.5, alpha: float = 1.0, inplace: bool = False) -> None:
        super().__init__()
        assert num_classes > 0, "Please provide a valid positive value for the num_classes."
        assert alpha > 0, "Alpha param can't be zero."

        self.num_classes = num_classes
        self.p = p
        self.alpha = alpha
        self.inplace = inplace

    def forward(self, batch: Tensor, target: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            batch (Tensor): Float tensor of size (B, C, H, W)
            target (Tensor): Integer tensor of size (B, )

        Returns:
            Tensor: Randomly transformed batch.
        """
        if batch.ndim != 4:
            raise ValueError(f"Batch ndim should be 4. Got {batch.ndim}")
        if target.ndim != 1:
            raise ValueError(f"Target ndim should be 1. Got {target.ndim}")
        if not batch.is_floating_point():
            raise TypeError(f"Batch dtype should be a float tensor. Got {batch.dtype}.")
        if target.dtype != torch.int64:
            raise TypeError(f"Target dtype should be torch.int64. Got {target.dtype}")

        if not self.inplace:
            batch = batch.clone()
            target = target.clone()

        if target.ndim == 1:
            target = torch.nn.functional.one_hot(target, num_classes=self.num_classes).to(dtype=batch.dtype)

        if torch.rand(1).item() >= self.p:
            return batch, target

        # It's faster to roll the batch by one instead of shuffling it to create image pairs
        batch_rolled = batch.roll(1, 0)
        target_rolled = target.roll(1, 0)

        # Implemented as on cutmix paper, page 12 (with minor corrections on typos).
        lambda_param = float(torch._sample_dirichlet(torch.tensor([self.alpha, self.alpha]))[0])
        W, H = torchvision.transforms.functional.get_image_size(batch)

        r_x = torch.randint(W, (1,))
        r_y = torch.randint(H, (1,))

        r = 0.5 * math.sqrt(1.0 - lambda_param)
        r_w_half = int(r * W)
        r_h_half = int(r * H)

        x1 = int(torch.clamp(r_x - r_w_half, min=0))
        y1 = int(torch.clamp(r_y - r_h_half, min=0))
        x2 = int(torch.clamp(r_x + r_w_half, max=W))
        y2 = int(torch.clamp(r_y + r_h_half, max=H))

        batch[:, :, y1:y2, x1:x2] = batch_rolled[:, :, y1:y2, x1:x2]
        lambda_param = float(1.0 - (x2 - x1) * (y2 - y1) / (W * H))

        target_rolled.mul_(1.0 - lambda_param)
        target.mul_(lambda_param).add_(target_rolled)

        return batch, target

    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}("
            f"num_classes={self.num_classes}"
            f", p={self.p}"
            f", alpha={self.alpha}"
            f", inplace={self.inplace}"
            f")"
        )
        return s


class ClassificationPresetTrain:
    def __init__(
            self,
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            interpolation=InterpolationMode.BILINEAR,
            hflip_prob=0.5,
            auto_augment_policy=None,
            random_erase_prob=0.0,
    ):
        trans = []
        if hflip_prob > 0:
            trans.append(transforms.RandomHorizontalFlip(hflip_prob))
        if auto_augment_policy is not None:
            if auto_augment_policy == "ra":
                trans.append(autoaugment.RandAugment(interpolation=interpolation))
            elif auto_augment_policy == "ta_wide":
                trans.append(autoaugment.TrivialAugmentWide(interpolation=interpolation))
            else:
                aa_policy = autoaugment.AutoAugmentPolicy(auto_augment_policy)
                trans.append(autoaugment.AutoAugment(policy=aa_policy, interpolation=interpolation))
        trans.extend(
            [
                transforms.PILToTensor(),
                transforms.ConvertImageDtype(torch.float),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
        if random_erase_prob > 0:
            trans.append(transforms.RandomErasing(p=random_erase_prob))

        self.transforms = transforms.Compose(trans)

    def __call__(self, img):
        return self.transforms(img)

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
        self.alpha = nn.Parameter(torch.tensor([6.], device=device))

        self.fc = nn.Linear(T, T)
        nn.init.constant_(self.fc.bias, -1)

        Lambda = self._generate_lambda_matrix()
        self.fc.weight.data = Lambda

    def _generate_lambda_matrix(self):
        Lambda = torch.zeros((self.T, self.T), device=self.device)
        for i in range(self.T):
            for j in range(i + 1):
                Lambda[i, j] = self.lam ** (i - j)
        return Lambda

    def forward(self, x_seq: torch.Tensor):
        batch_size, T, *feature_dims = x_seq.shape
        x_seq = x_seq.transpose(0, 1)

        # feature_size = torch.prod(torch.tensor(feature_dims)).item()
        # x_flat = x_seq.view(batch_size, T, feature_size)
        # x_flat_trans = x_flat.transpose(1, 2).reshape(feature_size * batch_size, T)
        #
        # h_seq = self.fc(x_flat_trans)
        h_seq = torch.addmm(self.fc.bias.unsqueeze(1), self.fc.weight, x_seq.flatten(1))
        spike = self.SurrogateSigmoid(h_seq, self.alpha)
        h_seq = torch.addmm(self.fc.bias.unsqueeze(1), self.fc.weight, x_seq.flatten(1) - spike) + spike
        spike = self.SurrogateSigmoid(h_seq, self.alpha * 3)
        h_seq = torch.addmm(self.fc.bias.unsqueeze(1), self.fc.weight, x_seq.flatten(1) - spike) + spike
        spike = self.SurrogateSigmoid(h_seq, self.alpha * 6)

        spike = spike.view(x_seq.shape).transpose(0, 1)
        spike = self.SurrogateSampling(spike)
        return spike


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
        self.alpha = torch.tensor([1.], device=device)

        self.fc = nn.Linear(T, T)
        nn.init.constant_(self.fc.bias, -1)

        Lambda = self._generate_lambda_matrix()
        self.fc.weight.data = Lambda

    def _generate_lambda_matrix(self):
        Lambda = torch.zeros((self.T, self.T), device=self.device)
        for i in range(self.T):
            for j in range(i + 1):
                Lambda[i, j] = self.lam ** (i - j)
        return Lambda

    def forward(self, x_seq: torch.Tensor):
        batch_size, T, *feature_dims = x_seq.shape
        x_seq = x_seq.transpose(0, 1)

        # feature_size = torch.prod(torch.tensor(feature_dims)).item()
        # x_flat = x_seq.view(batch_size, T, feature_size)
        # x_flat_trans = x_flat.transpose(1, 2).reshape(feature_size * batch_size, T)
        #
        # h_seq = self.fc(x_flat_trans)
        h_seq = torch.addmm(self.fc.bias.unsqueeze(1), torch.tril(self.fc.weight), x_seq.flatten(1))
        spike = self.SurrogateSigmoid(h_seq, self.alpha)
        h_seq = torch.addmm(self.fc.bias.unsqueeze(1), torch.tril(self.fc.weight), x_seq.flatten(1) - spike) + spike
        spike = self.SurrogateSigmoid(h_seq, self.alpha * 3)
        h_seq = torch.addmm(self.fc.bias.unsqueeze(1), torch.tril(self.fc.weight), x_seq.flatten(1) - spike) + spike
        spike = self.SurrogateSigmoid(h_seq, self.alpha * 12)

        spike = spike.view(x_seq.shape).transpose(0, 1)
        spike = self.SurrogateSampling(spike)
        return spike


class PSN(nn.Module):
    def __init__(self, T: int, surrogate_function: surrogate.SurrogateFunctionBase):
        super().__init__()
        self.surrogate_function = surrogate_function
        self.fc = nn.Linear(T, T)
        nn.init.constant_(self.fc.bias, -1)

    def forward(self, x_seq: torch.Tensor):
        # x_seq.shape = [T, N, *]
        batch_size, T, *feature_dims = x_seq.shape
        x_seq = x_seq.transpose(0, 1)

        # feature_size = torch.prod(torch.tensor(feature_dims)).item()
        # x_flat = x_seq.view(batch_size, T, feature_size)
        # x_flat_trans = x_flat.transpose(1, 2).reshape(feature_size * batch_size, T)
        #
        # h_seq = self.fc(x_flat_trans)
        h_seq = torch.addmm(self.fc.bias.unsqueeze(1), self.fc.weight, x_seq.flatten(1))
        spike = self.surrogate_function(h_seq).view(x_seq.shape).transpose(0, 1)
        return spike

class PSN0(nn.Module):
    def __init__(self, T: int, surrogate_function: surrogate.SurrogateFunctionBase, device=None):
        super().__init__()
        self.T = T
        self.surrogate_function = surrogate_function
        self.threshold = torch.tensor([1.0], device=device)
        self.max_iter = torch.tensor([2], device=device)
        self.lam = torch.tensor([0.5], device=device)
        self.device = device
        self.alpha = torch.tensor([12.], device=device)

        self.Lambda = self._generate_lambda_matrix()
        self.A = self.Lambda - torch.eye(self.T, device=self.device)

    def _generate_lambda_matrix(self):
        Lambda = torch.zeros((self.T, self.T), device=self.device)
        for i in range(self.T):
            for j in range(i + 1):
                Lambda[i, j] = self.lam ** (i - j)
        return Lambda

    def forward(self, x_seq: torch.Tensor):
        batch_size, T = x_seq.shape[:2]
        remaining_dims = x_seq.shape[2:]
        x_reshaped = x_seq.view(batch_size, T, -1)

        u = torch.matmul(self.Lambda, x_reshaped)
        o = self.surrogate_function(u - self.threshold).view(batch_size, T, *remaining_dims)

        return o

class IPSU(nn.Module):
    def __init__(self, T: int, surrogate_function: surrogate.SurrogateFunctionBase, device=None):
        super().__init__()
        self.surrogate_function = surrogate_function
        self.threshold = torch.tensor([1.0], device=device)
        self.max_iter = torch.tensor([2], device=device)
        self.lam = torch.tensor([0.5], device=device)
        self.T = T
        self.device = device

        self.fc = nn.Linear(T, T, bias=False)
        self.Lambda = self._generate_lambda_matrix()
        self.A = self.Lambda - torch.eye(self.T, device=self.device)

    def _generate_lambda_matrix(self):
        Lambda = torch.zeros((self.T, self.T), device=self.device)
        for i in range(self.T):
            for j in range(i + 1):
                Lambda[i, j] = self.lam ** (i - j)
        return Lambda

    def forward(self, x_seq: torch.Tensor):
        batch_size, T = x_seq.shape[:2]
        remaining_dims = x_seq.shape[2:]
        x_reshaped = x_seq.view(batch_size, T, -1)

        u = torch.matmul(self.Lambda, x_reshaped)
        u = torch.matmul(torch.tril(self.fc.weight), x_reshaped) - torch.matmul(self.A, u) - self.threshold
        spike = self.surrogate_function(u).view(batch_size, T, *remaining_dims)
        return spike

class LIF(nn.Module):
    def __init__(self, T: int, surrogate_function: surrogate.SurrogateFunctionBase, device=None):
        super().__init__()
        self.T = T
        # self.surrogate_function = StochasticST.apply
        self.surrogate_function = surrogate_function
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


def create_neuron(neu: str, **kwargs):
    if neu == 'PSN0':
        return PSN0(T=kwargs['T'], surrogate_function=kwargs['surrogate_function'], device=kwargs['device'])
    elif neu == 'PSN':
        return PSN(T=kwargs['T'], surrogate_function=kwargs['surrogate_function'])
    elif neu == 'FPT_learnable_alpha':
        return FPT_learnable_alpha(T=kwargs['T'], surrogate_function=kwargs['surrogate_function'], device=kwargs['device'])
    elif neu == 'LIF':
        return LIF(T=kwargs['T'], surrogate_function=kwargs['surrogate_function'], device=kwargs['device'])
    elif neu == 'IPSU':
        return IPSU(T=kwargs['T'], surrogate_function=kwargs['surrogate_function'], device=kwargs['device'])
    elif neu == 'FPT_adaptive_alpha':
        return FPT_adaptive_alpha(T=kwargs['T'], surrogate_function=kwargs['surrogate_function'], device=kwargs['device'])
    return None


class CIFAR10Net(nn.Module):
    def __init__(self, channels, neu: str, T: int, class_num: int, P: int = -1, exp_init: bool = False,
                 device: str = None):
        super().__init__()
        conv = []
        for i in range(2):
            for j in range(3):
                if conv.__len__() == 0:
                    in_channels = 3
                else:
                    in_channels = channels
                conv.append(layer.Conv1d(in_channels, channels, kernel_size=3, padding=1, bias=False))
                conv.append(layer.BatchNorm1d(channels))
                conv.append(
                    create_neuron(neu, T=T, features=channels, surrogate_function=surrogate.ATan(), channels=channels,
                                  P=P, exp_init=exp_init, device=device))

            conv.append(layer.AvgPool1d(2))

        self.conv = nn.Sequential(*conv)

        self.fc = nn.Sequential(
            layer.Flatten(),
            layer.Linear(channels * 8, channels * 8 // 4),
            create_neuron(neu, T=T, features=channels * 8 // 4, surrogate_function=surrogate.ATan(), P=P,
                          exp_init=exp_init, device=device),
            layer.Linear(channels * 8 // 4, class_num),
        )

        functional.set_step_mode(self, 'm')

    def forward(self, x_seq: torch.Tensor):
        # [N, C, H, W] -> [W, N, C, H]
        x_seq = x_seq.permute(0, 3, 1, 2)
        x_seq = self.fc(self.conv(x_seq))  # [W, N, C]
        return x_seq.mean(1)


def main():
    parser = argparse.ArgumentParser(description='Classify Sequential CIFAR10/100')
    parser.add_argument('-device', default='cuda:3', help='device')
    parser.add_argument('-b', default=128, type=int, help='batch size')
    parser.add_argument('-epochs', default=256, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-j', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-data-dir', type=str, default='/data/Datasets/cifar-100-python', help='root dir of CIFAR10/100 dataset')
    parser.add_argument('-out-dir', type=str, default='./logs_scf100', help='root dir for saving logs and checkpoint')
    parser.add_argument('-resume', type=str, help='resume from the checkpoint path')
    parser.add_argument('-amp', default=True, help='automatic mixed precision training')
    parser.add_argument('-opt', type=str, default='sgd', help='use which optimizer. sgd or adamw')
    parser.add_argument('-momentum', default=0.9, type=float, help='momentum for SGD')
    parser.add_argument('-lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('-channels', default=128, type=int, help='channels of CSNN')
    parser.add_argument('-neu', type=str, default='FPT_adaptive_alpha', help='use which neuron')
    parser.add_argument('-class-num', type=int, default=100)

    parser.add_argument('-P', type=int, default=None, help='the order of the masked/sliding PSN')
    parser.add_argument('-exp-init', default=True,
                        help='use the exp init method to initialize the weight of SPSN')

    args = parser.parse_args()
    print(args)

    mixup_transforms = []
    mixup_transforms.append(RandomMixup(args.class_num, p=1., alpha=0.2))
    mixup_transforms.append(RandomCutmix(args.class_num, p=1.0, alpha=1.))
    mixupcutmix = torchvision.transforms.RandomChoice(mixup_transforms)
    collate_fn = lambda batch: mixupcutmix(*default_collate(batch))  # noqa: E731

    if args.class_num == 100:
        transform_train = ClassificationPresetTrain(mean=(0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                                                    std=(0.2673342858792401, 0.2564384629170883, 0.27615047132568404),
                                                    interpolation=InterpolationMode('bilinear'),
                                                    auto_augment_policy='ta_wide',
                                                    random_erase_prob=0.1)
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                                 (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)),
        ])

    if args.class_num == 100:
        train_set = torchvision.datasets.CIFAR100(
            root=args.data_dir,
            train=True,
            transform=transform_train,
            download=True)

        test_set = torchvision.datasets.CIFAR100(
            root=args.data_dir,
            train=False,
            transform=transform_test,
            download=True)

    train_data_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=args.b,
        collate_fn=collate_fn,
        shuffle=True,
        drop_last=True,
        num_workers=args.j,
        pin_memory=True
    )

    test_data_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=args.b,
        shuffle=False,
        drop_last=False,
        num_workers=args.j,
        pin_memory=True
    )
    out_dir = f'{args.neu}_e{args.epochs}_b{args.b}_{args.opt}_lr{args.lr}_c{args.channels}'
    if args.amp:
        out_dir += '_amp'

    if args.P is not None:
        out_dir += f'_P{args.P}'
        if args.exp_init:
            out_dir += '_ei'

    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    out_dir += f'_{timestamp}'

    pt_dir = os.path.join(args.out_dir, 'pt', out_dir)
    out_dir = os.path.join(args.out_dir, out_dir)

    if not os.path.exists(pt_dir):
        os.makedirs(pt_dir)

    net = CIFAR10Net(channels=args.channels, neu=args.neu, T=32, class_num=args.class_num, P=args.P,
                     exp_init=args.exp_init, device=args.device)
    net.to(args.device)

    scaler = None
    if args.amp:
        scaler = amp.GradScaler()

    start_epoch = 0
    max_test_acc = -1

    optimizer = None
    if args.opt == 'sgd':
        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)
    elif args.opt == 'adamw':
        optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.)
    else:
        raise NotImplementedError(args.opt)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        net.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        max_test_acc = checkpoint['max_test_acc']
        print(max_test_acc)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print(f'Mkdir {out_dir}.')

    writer = SummaryWriter(out_dir, purge_step=start_epoch)

    with open(os.path.join(out_dir, 'args.txt'), 'w', encoding='utf-8') as args_txt:
        args_txt.write(str(args))
        args_txt.write('\n')
        args_txt.write(' '.join(sys.argv))

    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()
        net.train()
        train_loss = 0
        train_acc = 0
        train_samples = 0

        for batch_index, (img, label) in enumerate(train_data_loader):
            optimizer.zero_grad()
            img = img.to(args.device, non_blocking=True)
            label = label.to(args.device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=scaler is not None):
                y = net(img)
                loss = F.cross_entropy(y, label, label_smoothing=0.1)

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            train_samples += label.shape[0]
            train_loss += loss.item() * label.shape[0]
            train_acc += (y.argmax(1) == label.argmax(1)).float().sum().item()

            functional.reset_net(net)

        train_time = time.time()
        train_speed = train_samples / (train_time - start_time)
        train_loss /= train_samples
        train_acc /= train_samples

        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('train_acc', train_acc, epoch)
        lr_scheduler.step()

        net.eval()
        test_loss = 0
        test_acc = 0
        test_samples = 0
        with torch.no_grad():
            for img, label in test_data_loader:
                img = img.to(args.device)
                label = label.to(args.device)
                y = net(img)
                loss = F.cross_entropy(y, label)
                test_samples += label.numel()
                test_loss += loss.item() * label.numel()
                test_acc += (y.argmax(1) == label).float().sum().item()
                functional.reset_net(net)
        test_time = time.time()
        test_speed = test_samples / (test_time - train_time)
        test_loss /= test_samples
        test_acc /= test_samples
        writer.add_scalar('test_loss', test_loss, epoch)
        writer.add_scalar('test_acc', test_acc, epoch)

        save_max = False
        if test_acc > max_test_acc:
            max_test_acc = test_acc
            save_max = True

        checkpoint = {
            'net': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            'max_test_acc': max_test_acc
        }

        if save_max:
            torch.save(checkpoint, os.path.join(pt_dir, 'checkpoint_max.pth'))

        torch.save(checkpoint, os.path.join(pt_dir, 'checkpoint_latest.pth'))

        print(args)
        print(out_dir)
        print(
            f'epoch = {epoch}, train_loss ={train_loss: .4f}, train_acc ={train_acc: .4f}, test_loss ={test_loss: .4f}, test_acc ={test_acc: .4f}, max_test_acc ={max_test_acc: .4f}')
        print(f'train speed ={train_speed: .4f} images/s, test speed ={test_speed: .4f} images/s')
        print(
            f'escape time = {(datetime.datetime.now() + datetime.timedelta(seconds=(time.time() - start_time) * (args.epochs - epoch))).strftime("%Y-%m-%d %H:%M:%S")}\n')


if __name__ == '__main__':
    main()
