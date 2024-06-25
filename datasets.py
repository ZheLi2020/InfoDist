from torch.utils.data import Dataset
import numpy as np
import torch
from PIL import Image
import os
from absl import logging


class MedSyn(Dataset):
    def __init__(self, img_path, transform):
        super().__init__()
        self.total_files = _list_image_files_recursively(img_path)
        self.class_names = [os.path.basename(path).split("_")[0] for path in self.total_files]
        logging.info('Prepare train dataset done')

        self.targets = [int(x) for x in self.class_names]
        self.transform = transform

    def __getitem__(self, idx):
        path = self.total_files[idx]
        img = Image.open(path)
        img = img.convert('RGB')
        img = torch.from_numpy(np.array(img)/255).type(torch.FloatTensor)

        return img.permute((2, 0, 1)), self.targets[idx]

    def __len__(self):
        return len(self.total_files)


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

