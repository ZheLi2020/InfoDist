import torch
import numpy as np
from tqdm import tqdm

from cluster_methods import querybykmeans, querybyumap, querybyimageumap
from cluster_infomap import querybyumapinfouniform_high, querybyembedinfo_high
from absl import logging


def build_dataset(ds, class_map, num_classes):
    images_all = []
    labels_all = []
    indices_class = [[] for c in range(num_classes)]
    logging.info("BUILDING DATASET")
    idx = 0
    for i in tqdm(range(len(ds))):
        sample = ds[i]
        images_all.append(torch.unsqueeze(sample[0], dim=0))
        labels_all.append(class_map[torch.tensor(sample[1]).item()])

        # idx += 1
        # if idx > 1600:
        #     break
    for i, lab in tqdm(enumerate(labels_all)):
        indices_class[lab].append(i)
    images_all = torch.cat(images_all, dim=0).to("cpu")
    labels_all = torch.tensor(labels_all, dtype=torch.long, device="cpu")

    indices_class = [np.array(ele) for ele in indices_class if ele != []]

    return images_all, labels_all, indices_class


def choose_dataset(images_all, labels_all, indices_class, num_classes, args):
    logging.info(f'choose {args.num_cluster * args.subsample} images by {args.cluster} from each class')
    images_small = []
    labels_small = []
    indices_class_small = [[] for c in range(num_classes)]
    for n in range(num_classes):
        if args.cluster == 'kmeans':
            n_idx = querybykmeans(images_all[indices_class[n]], num_classes, args)
        elif args.cluster == 'umap':
            n_idx = querybyumap(images_all[indices_class[n]], num_classes, args)
        elif args.cluster == 'imageumap':
            n_idx = querybyimageumap(images_all[indices_class[n]], args)
        elif args.cluster == 'umapinfounihigh':
            n_idx = querybyumapinfouniform_high(images_all[indices_class[n]], args)
        elif args.cluster == 'embedinfo':
            n_idx = querybyembedinfo_high(images_all[indices_class[n]], num_classes, args)
        else:
            n_idx = np.random.permutation(range(indices_class[n].shape[0]))[:args.num_cluster * args.subsample]
        img_idx = indices_class[n][n_idx]
        images_small.append(images_all[img_idx])
        labels_small += labels_all[img_idx].tolist()

    for i, lab in tqdm(enumerate(labels_small)):
        indices_class_small[lab].append(i)
    images_small = torch.cat(images_small, dim=0).to("cpu")
    labels_small = torch.tensor(labels_small, dtype=torch.long, device="cpu")

    indices_class_small = [np.array(ele) for ele in indices_class_small if ele != []]

    return images_small, labels_small, indices_class_small


def choose_random(images_all, labels_all, indices_class, num_classes, args):
    num_imgs = args.num_cluster * args.subsample
    logging.info(f'choose {num_imgs} images randomly from each class')
    images_small = []
    labels_small = []
    indices_class_small = [[] for c in range(num_classes)]
    for c in range(num_classes):
        idx_shuffle = np.random.permutation(indices_class[c])[:num_imgs]
        images_small.append(images_all[idx_shuffle])
        labels_small += labels_all[idx_shuffle].tolist()

    for i, lab in tqdm(enumerate(labels_small)):
        indices_class_small[lab].append(i)
    images_small = torch.cat(images_small, dim=0).to("cpu")
    labels_small = torch.tensor(labels_small, dtype=torch.long, device="cpu")

    indices_class_small = [np.array(ele) for ele in indices_class_small if ele != []]

    return images_small, labels_small, indices_class_small


def get_eval_lrs(args):
    eval_pool_dict = {
        "ConvNet": 0.001,
        "ConvNet_BN": 0.001,
        "ResNet18": 0.001,
        "VGG11": 0.0001,
        "AlexNet": 0.001,
        "ViT": 0.001,

        "AlexNetCIFAR": 0.001,
        "ResNet18CIFAR": 0.001,
        "VGG11CIFAR": 0.0001,
        "ViTCIFAR": 0.001,
    }

    return eval_pool_dict

