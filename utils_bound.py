import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional
import os
import tqdm
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from ema_pytorch import EMA
from networks import *
from absl import logging
from medmnist import PathMNIST
from datasets import MedSyn
from PIL import Image
from metrics import calculate_auc_and_fscore
from contrastive_loss import get_logp_boundary, calculate_bg_spp_loss


class Config:
    medmnist = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    dict = {
        "medreal": medmnist,
        "medsyn": medmnist
    }

    mean = torch.tensor([0.4377, 0.4438, 0.4728]).reshape(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)


config = Config()


def get_dataset(dataset, data_path, batch_size=1, res=None, args=None):

    if dataset.startswith("medreal"):
        channel = 3
        im_size = (res, res)
        num_classes = 9
        config.img_net_classes = config.dict[dataset]

        mean = [0.741, 0.533, 0.706]
        std = [0.402, 0.821, 0.407]
        # mean = [189, 136, 180]
        # std = [995, 2032, 1007]
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=mean, std=std),
                                        transforms.Resize(res, interpolation=transforms.InterpolationMode.BICUBIC),
                                        transforms.CenterCrop(res)
                                        ])

        dst_train = PathMNIST(split='train', download=True)
        dst_train.transform = transform
        loader_train_dict = torch.utils.data.DataLoader(dst_train, batch_size=batch_size, shuffle=True, num_workers=16)

        dst_test = PathMNIST(split='test', download=True)
        dst_test.transform = transform

        class_map = {x: i for i, x in enumerate(config.img_net_classes)}
        class_map_inv = {i: x for i, x in enumerate(config.img_net_classes)}
        class_names = None

    elif dataset.startswith("medsyn"):
        channel = 3
        im_size = (res, res)
        num_classes = 9
        config.img_net_classes = config.dict[dataset]

        mean = [0.741, 0.533, 0.706]
        std = [0.402, 0.821, 0.407]

        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=mean, std=std),
                                        transforms.Resize(res),
                                        transforms.CenterCrop(res)
                                        ])

        dst_train = MedSyn(data_path, transform=transform)
        loader_train_dict = torch.utils.data.DataLoader(dst_train, batch_size=batch_size, shuffle=True, num_workers=16)

        dst_test = PathMNIST(split='test', download=True)
        dst_test.transform = transform

        class_map = {x: i for i, x in enumerate(config.img_net_classes)}
        class_map_inv = {i: x for i, x in enumerate(config.img_net_classes)}
        class_names = None

    else:
        exit('unknown dataset: %s'%dataset)

    testloader = torch.utils.data.DataLoader(dst_test, batch_size=args.batch_test, shuffle=False, num_workers=0)

    return channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader, loader_train_dict, class_map, class_map_inv


class TensorDataset(Dataset):
    def __init__(self, images, labels): # images: n x c x h x w tensor
        self.images = images.detach().float()
        self.labels = labels.detach()

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return self.images.shape[0]


def get_network(model, channel, num_classes, im_size=(32, 32), depth=3, width=128, norm="instancenorm", args=None):
    torch.random.manual_seed(int(time.time() * 1000) % 100000)

    if model == 'AlexNet':
        net = AlexNet(channel, num_classes=num_classes, im_size=im_size)
    elif model == 'VGG11':
        net = VGG11(channel=channel, num_classes=num_classes)
    elif model == 'ResNet18':
        net = ResNet18(channel=channel, num_classes=num_classes, norm=norm)
    elif model == "ViT":
        net = ViT(image_size=im_size, patch_size=16, num_classes=num_classes, dim=512, depth=10,
                  heads=8, mlp_dim=512, dropout=0.1, emb_dropout=0.1,)

    elif model == "ConvNet":
        net = ConvNet(channel, num_classes, net_width=width, net_depth=depth, net_act='relu', net_norm=norm, im_size=im_size)

    else:
        net = None
        exit('DC error: unknown model')

    if args.distributed:
        gpu_num = len(args.device_ids)
        if gpu_num>0:
            # device = args.device
            if gpu_num>1:
                net = nn.DataParallel(net, device_ids=args.device_ids)
        else:
            device = 'cpu'
        net = net.to(args.device)

    return net


def get_time():
    return str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))


def epoch(mode, dataloader, net, optimizer, criterion, args, aug, num_classes=9):
    loss_avg, acc_avg, num_exp = 0, 0, 0
    net = net.to(args.device)
    class_map = {x: i for i, x in enumerate(config.img_net_classes)}

    if mode == 'train':
        net.train()
    else:
        net.eval()

    label_list = []
    pred_list = []
    prob_list = []
    for i_batch, datum in enumerate(dataloader):
        img = datum[0].to(args.device)
        lab = datum[1].to(args.device)
        lab = torch.tensor([class_map[x.item()] for x in lab]).to(args.device)

        if aug:
            if args.dsa:
                img = DiffAugment(img, args.dsa_strategy, param=args.dsa_param)
            else:
                img = augment(img, args.dc_aug_param, device=args.device)

        n_b = lab.shape[0]

        output = net(img)
        # print(output)

        loss = criterion(output, lab)

        predicted = torch.argmax(output.data, 1)
        correct = (predicted == lab).sum()

        loss_avg += loss.item()*n_b
        acc_avg += correct.item()
        num_exp += n_b

        # sklearn
        label_list.append(lab.detach().cpu().numpy())  # .detach().cpu().numpy()
        pred_list.append(predicted.detach().cpu().numpy())
        prob_list.append(output.softmax(dim=-1).detach().cpu().numpy())

        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    loss_avg /= num_exp
    acc_avg /= num_exp

    auc_score, fscore = calculate_auc_and_fscore(pred_list, prob_list, label_list, num_classes)

    return loss_avg, acc_avg, auc_score, fscore


def epoch_contrastive(mode, dataloader, net, optimizer, criterion, args, aug, num_classes=9):
    loss_avg, acc_avg, num_exp = 0, 0, 0
    net = net.to(args.device)
    class_map = {x: i for i, x in enumerate(config.img_net_classes)}

    if mode == 'train':
        net.train()
    else:
        net.eval()

    label_list = []
    pred_list = []
    prob_list = []
    loss = 0
    for i_batch, datum in enumerate(dataloader):
        img = datum[0].to(args.device)
        lab = datum[1].to(args.device)
        lab = torch.tensor([class_map[x.item()] for x in lab]).to(args.device)
        n_b = lab.shape[0]

        if aug:
            if args.dsa:
                img = DiffAugment(img, args.dsa_strategy, param=args.dsa_param)
            else:
                img = augment(img, args.dc_aug_param, device=args.device)

        output = net(img)
        # add softmax
        output = F.softmax(output, dim=1)
        loss_con = 0
        for c in range(num_classes):
            mask = lab == c
            if mask.sum() > 0:
                boundaries = get_logp_boundary(output[:, c], mask, args.pos_beta, args.margin_tau)
                loss_c_con = calculate_bg_spp_loss(output[:, c], mask, boundaries)
                loss_con += loss_c_con

        loss_ce = criterion(output, lab)
        loss = loss_ce + loss_con

        predicted = torch.argmax(output.data, 1)
        correct = (predicted == lab).sum()

        loss_avg += loss.item() * n_b
        acc_avg += correct.item()
        num_exp += n_b

        # sklearn
        label_list.append(lab.detach().cpu().numpy())  # .detach().cpu().numpy()
        pred_list.append(predicted.detach().cpu().numpy())
        prob_list.append(output.softmax(dim=-1).detach().cpu().numpy())

        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    loss_avg /= num_exp
    acc_avg /= num_exp

    auc_score, fscore = calculate_auc_and_fscore(pred_list, prob_list, label_list, num_classes)

    return loss_avg, acc_avg, auc_score, fscore


def train_synset(it_run, net, images_train, labels_train, testloader, args, decay="cosine", aug=True):
    net = net.to(args.device)
    # images_train = images_train.to(args.device)
    # labels_train = labels_train.to(args.device)
    lr = float(args.lr_net)
    Epoch = int(args.total_epoch)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)

    if decay == "cosine":
        sched1 = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.0000001, end_factor=1.0, total_iters=Epoch//2)
        sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Epoch//2)

    elif decay == "step":
        lmbda1 = lambda epoch: 1.0
        sched1 = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lmbda1)
        lmbda2 = lambda epoch: 0.1
        sched2 = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lmbda2)

    sched = sched1

    ema = EMA(net, beta=0.995, power=1, update_after_step=0, update_every=1)
    criterion = nn.CrossEntropyLoss().to(args.device)
    if args.contrastive:
        logging.info('train with contrastive loss')

    dst_train = TensorDataset(images_train, labels_train)
    trainloader = torch.utils.data.DataLoader(dst_train, batch_size=args.batch_train, shuffle=True, num_workers=0)

    start = time.time()
    acc_train_list = []
    loss_train_list = []
    for ep in tqdm.tqdm(range(Epoch)):
        if args.contrastive:
            loss_train, acc_train, _, _ = epoch_contrastive('train', trainloader, net, optimizer, criterion, args, aug=aug)
        else:
            loss_train, acc_train, _, _ = epoch('train', trainloader, net, optimizer, criterion, args, aug=aug)

        acc_train_list.append(acc_train)
        loss_train_list.append(loss_train)
        ema.update()
        sched.step()
        if ep == Epoch // 2:
            sched = sched2

    with torch.no_grad():
        if args.contrastive:
            loss_test, acc_test, auc_test, fscore_test = epoch_contrastive('test', testloader, ema, optimizer, criterion, args, aug=False)
        else:
            loss_test, acc_test, auc_test, fscore_test = epoch('test', testloader, ema, optimizer, criterion, args,
                                                               aug=False)

    time_train = time.time() - start

    logging.info(f'Evaluate_{it_run}: epoch = {Epoch} train time = {int(time_train)} s '
                 f'train loss = {loss_train} train acc = {acc_train}, test acc = {acc_test}')
    logging.info(f'Evaluate_{it_run}: AUC = {auc_test}, Fscore = {fscore_test}')

    return net, acc_test, auc_test, fscore_test


def get_eval_pool(eval_mode, model, model_eval):
    if eval_mode == 'M':  # multiple architectures
        model_eval_pool = [model, "ResNet18"]
    elif eval_mode == 'R':  # multiple architectures
        model_eval_pool = [model, "ResNet18", "VGG11", "AlexNet", "ViT"]
    else:
        model_eval_pool = [model_eval]
    return model_eval_pool


class ParamDiffAug():
    def __init__(self):
        self.aug_mode = 'S' #'multiple or single'
        self.prob_flip = 0.5
        self.ratio_scale = 1.2
        self.ratio_rotate = 15.0
        self.ratio_crop_pad = 0.125
        self.ratio_cutout = 0.5 # the size would be 0.5x0.5
        self.ratio_noise = 0.05
        self.brightness = 1.0
        self.saturation = 2.0
        self.contrast = 0.5


def set_seed_DiffAug(param):
    if param.latestseed == -1:
        return
    else:
        torch.random.manual_seed(param.latestseed)
        param.latestseed += 1


def DiffAugment(x, strategy='', seed = -1, param = None):
    if seed == -1:
        param.batchmode = False
    else:
        param.batchmode = True

    param.latestseed = seed

    if strategy == 'None' or strategy == 'none':
        return x

    if strategy:
        if param.aug_mode == 'M': # original
            for p in strategy.split('_'):
                for f in AUGMENT_FNS[p]:
                    x = f(x, param)
        elif param.aug_mode == 'S':
            pbties = strategy.split('_')
            set_seed_DiffAug(param)
            p = pbties[torch.randint(0, len(pbties), size=(1,)).item()]
            for f in AUGMENT_FNS[p]:
                x = f(x, param)
        else:
            exit('Error ZH: unknown augmentation mode.')
        x = x.contiguous()
    return x


# We implement the following differentiable augmentation strategies based on the code provided in https://github.com/mit-han-lab/data-efficient-gans.
def rand_scale(x, param):
    ratio = param.ratio_scale
    set_seed_DiffAug(param)
    sx = torch.rand(x.shape[0]) * (ratio - 1.0/ratio) + 1.0/ratio
    set_seed_DiffAug(param)
    sy = torch.rand(x.shape[0]) * (ratio - 1.0/ratio) + 1.0/ratio
    theta = [[[sx[i], 0,  0],
            [0,  sy[i], 0],] for i in range(x.shape[0])]
    theta = torch.tensor(theta, dtype=torch.float)
    if param.batchmode: # batch-wise:
        theta[:] = theta[0].clone()
    grid = F.affine_grid(theta, x.shape, align_corners=True).to(x.device)
    x = F.grid_sample(x, grid, align_corners=True)
    return x


def rand_rotate(x, param): # [-180, 180], 90: anticlockwise 90 degree
    ratio = param.ratio_rotate
    set_seed_DiffAug(param)
    theta = (torch.rand(x.shape[0]) - 0.5) * 2 * ratio / 180 * float(np.pi)
    theta = [[[torch.cos(theta[i]), torch.sin(-theta[i]), 0],
        [torch.sin(theta[i]), torch.cos(theta[i]),  0],]  for i in range(x.shape[0])]
    theta = torch.tensor(theta, dtype=torch.float)
    if param.batchmode: # batch-wise:
        theta[:] = theta[0].clone()
    grid = F.affine_grid(theta, x.shape, align_corners=True).to(x.device)
    x = F.grid_sample(x, grid, align_corners=True)
    return x


def rand_flip(x, param):
    prob = param.prob_flip
    set_seed_DiffAug(param)
    randf = torch.rand(x.size(0), 1, 1, 1, device=x.device)
    if param.batchmode: # batch-wise:
        randf[:] = randf[0].clone()
    return torch.where(randf < prob, x.flip(3), x)


def rand_brightness(x, param):
    ratio = param.brightness
    set_seed_DiffAug(param)
    randb = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
    if param.batchmode:  # batch-wise:
        randb[:] = randb[0].clone()
    x = x + (randb - 0.5)*ratio
    return x


def rand_saturation(x, param):
    ratio = param.saturation
    x_mean = x.mean(dim=1, keepdim=True)
    set_seed_DiffAug(param)
    rands = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
    if param.batchmode:  # batch-wise:
        rands[:] = rands[0].clone()
    x = (x - x_mean) * (rands * ratio) + x_mean
    return x


def rand_contrast(x, param):
    ratio = param.contrast
    x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
    set_seed_DiffAug(param)
    randc = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
    if param.batchmode:  # batch-wise:
        randc[:] = randc[0].clone()
    x = (x - x_mean) * (randc + ratio) + x_mean
    return x


def rand_crop(x, param):
    # The image is padded on its surrounding and then cropped.
    ratio = param.ratio_crop_pad
    shift_x, shift_y = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    set_seed_DiffAug(param)
    translation_x = torch.randint(-shift_x, shift_x + 1, size=[x.size(0), 1, 1], device=x.device)
    set_seed_DiffAug(param)
    translation_y = torch.randint(-shift_y, shift_y + 1, size=[x.size(0), 1, 1], device=x.device)
    if param.batchmode:  # batch-wise:
        translation_x[:] = translation_x[0].clone()
        translation_y[:] = translation_y[0].clone()
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(x.size(2), dtype=torch.long, device=x.device),
        torch.arange(x.size(3), dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
    grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
    x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
    x = x_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2)
    return x


def rand_cutout(x, param):
    ratio = param.ratio_cutout
    cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    set_seed_DiffAug(param)
    offset_x = torch.randint(0, x.size(2) + (1 - cutout_size[0] % 2), size=[x.size(0), 1, 1], device=x.device)
    set_seed_DiffAug(param)
    offset_y = torch.randint(0, x.size(3) + (1 - cutout_size[1] % 2), size=[x.size(0), 1, 1], device=x.device)
    if param.batchmode:  # batch-wise:
        offset_x[:] = offset_x[0].clone()
        offset_y[:] = offset_y[0].clone()
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
        torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1)
    grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1)
    mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    mask[grid_batch, grid_x, grid_y] = 0
    x = x * mask.unsqueeze(1)
    return x


AUGMENT_FNS = {
    'color': [rand_brightness, rand_saturation, rand_contrast],
    'crop': [rand_crop],
    'cutout': [rand_cutout],
    'flip': [rand_flip],
    'scale': [rand_scale],
    'rotate': [rand_rotate],
}