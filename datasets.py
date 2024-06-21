from torch.utils.data import Dataset
import numpy as np
import torch
from PIL import Image
import os
from absl import logging
import random


class UnlabeledDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        data = tuple(self.dataset[item][:-1])  # remove label
        if len(data) == 1:
            data = data[0]
        return data


class DatasetFactory(object):

    def __init__(self):
        self.train = None
        self.test = None

    def get_split(self, split, labeled=False):
        if split == "train":
            dataset = self.train
        elif split == "test":
            dataset = self.test
        else:
            raise ValueError

        if self.has_label:
            return dataset if labeled else UnlabeledDataset(dataset)
        else:
            assert not labeled
            return dataset

    def unpreprocess(self, v):  # to B C H W and [0, 1]
        v = 0.5 * (v + 1.)
        v.clamp_(0., 1.)
        return v

    @property
    def has_label(self):
        return True

    @property
    def data_shape(self):
        raise NotImplementedError

    @property
    def data_dim(self):
        return int(np.prod(self.data_shape))

    @property
    def fid_stat(self):
        return None

    def sample_label(self, n_samples, device):
        raise NotImplementedError

    def label_prob(self, k):
        raise NotImplementedError


class Camelyon(DatasetFactory):  # the moments calculated by Stable Diffusion image encoder
    def __init__(self, path, transform, split):
        super().__init__()
        self.split = split
        if split == 'train':
            self.normal_files = _list_image_files_recursively(os.path.join(path, 'camelyon16/normal'))
            self.tumor_files = _list_image_files_recursively(os.path.join(path, 'camelyon16/tumor'))

            self.total_files = self.normal_files + self.tumor_files
            self.class_names = [os.path.basename(path).split("_")[0] for path in self.total_files]
            logging.info('Prepare train dataset done')

        elif split == 'test':
            self.total_files = _list_image_files_recursively(os.path.join(path, 'camelyon16/test'))

            tumor_mask_paths = '../glad_histo/data/camelyon16/test_mask'
            test_tumor_paths = os.listdir(tumor_mask_paths)
            test_tumor_name = [sst.split('.')[0] for sst in test_tumor_paths]
            self.class_names = []
            for nn in self.total_files:
                nn_base = os.path.basename(nn).split("-")[0]
                if nn_base in test_tumor_name:
                    self.class_names.append('tumor')
                else:
                    self.class_names.append('normal')

            logging.info('Prepare test dataset done')

        self.class_to_idx = {x: i for i, x in enumerate(sorted(set(self.class_names)))}
        self.targets = [self.class_to_idx[x] for x in self.class_names]
        self.transform = transform

    def __getitem__(self, idx):
        path = self.total_files[idx]
        img = Image.open(path)
        img = img.convert('RGB')
        img = torch.from_numpy(np.array(img)).type(torch.FloatTensor)

        return img.permute((2, 0, 1)), self.targets[idx]

    @property
    def data_shape(self):
        return 3, 256, 256

    @property
    def fid_stat(self):
        return f'assets/camel_fid_stats/fid_stats_camel256_32.npz'

    def sample_label(self, n_samples, device):
        return torch.randint(0, 2, (n_samples,), device=device)  # number of classes 1000


class MedSyn(Dataset):  # the moments calculated by Stable Diffusion image encoder
    def __init__(self, img_path, transform):
        super().__init__()
        self.total_files = _list_image_files_recursively(img_path)
        self.class_names = [os.path.basename(path).split("_")[0] for path in self.total_files]
        logging.info('Prepare train dataset done')

        self.targets = [int(x) for x in self.class_names]
        #self.transform = transform

    def __getitem__(self, idx):
        path = self.total_files[idx]
        img = Image.open(path)
        img = img.convert('RGB')
        img = torch.from_numpy(np.array(img)/255).type(torch.FloatTensor)

        return img.permute((2, 0, 1)), self.targets[idx]

    def __len__(self):
        return len(self.total_files)

    @property
    def data_shape(self):
        return 3, 256, 256

    @property
    def fid_stat(self):
        return f'assets/camel_fid_stats/fid_stats_camel256_32.npz'

    def sample_label(self, n_samples, device):
        return torch.randint(0, 2, (n_samples,), device=device)  # number of classes 1000


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(os.listdir(data_dir)):
        full_path = os.path.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "npy"]:
            results.append(full_path)
        elif os.listdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results

