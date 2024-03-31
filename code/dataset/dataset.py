import os
import pickle

import cv2
import mrcfile
import numpy as np
import rich
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from skimage.draw import disk
from torch.utils.data import DataLoader, Dataset

from ..utils import compute_ctf


class EMPIARDataset(Dataset):
    def __init__(self, mrcs: str, ctf: str, poses: str, size=256, sign=1) -> None:
        super().__init__()
        self.size = size

        with open(poses, "rb") as f:
            poses = pickle.load(f)
        self.rotations, self.translations = poses

        with open(ctf, "rb") as f:
            self.ctf_params = pickle.load(f)

        with mrcfile.open(mrcs) as f:
            self.images = sign * f.data

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index) -> dict:
        sample = {}

        sample["rotations"] = torch.from_numpy(self.rotations[index]).float()
        sample["translations"] = torch.from_numpy(np.concatenate([self.translations[index], np.array([0])])).float()
        sample["images"] = torch.from_numpy(cv2.resize(self.images[index].copy(),
                                                       (self.size, self.size), interpolation=cv2.INTER_LINEAR)).float()
        sample["ctf_params"] = torch.from_numpy(self.ctf_params[index]).float()

        freq_v = np.fft.fftshift(np.fft.fftfreq(self.size))
        freq_h = np.fft.fftshift(np.fft.fftfreq(self.size))
        freqs = torch.from_numpy(np.stack([freq.flatten() for freq in np.meshgrid(freq_v, freq_h, indexing="ij")],
                                          axis=1)) / (sample["ctf_params"][1] * sample["ctf_params"][0] / self.size)

        rr, cc = disk((self.size // 2, self.size // 2), self.size // 2)
        freqs_mask = np.zeros((self.size, self.size))
        freqs_mask[rr, cc] = 1
        sample["freqs_mask"] = torch.from_numpy(freqs_mask).float()
        sample["ctfs"] = compute_ctf(freqs, *torch.split(sample["ctf_params"][2:], 1, 0)).reshape(sample["images"].shape).float() * sample["freqs_mask"]

        return sample
