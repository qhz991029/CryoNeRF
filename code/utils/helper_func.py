import numpy as np
import torch


def to_cuda(sample: dict):
    cuda_sample = {}
    for k, v in sample.items():
        if type(v) == torch.Tensor:
            cuda_sample[k] = v.cuda()
        else:
            cuda_sample[k] = v

    return cuda_sample


def positional_encoding(coords: torch.Tensor, enc_dim=10):
    '''
    coords (B, N, 3): The 3D coordinates of position to query.
    '''
    # print(coords.requires_grad)
    freqs = torch.exp2(torch.arange(enc_dim)) / 10
    freqs = freqs.repeat_interleave(2).reshape(1, 1, 1, 1, 2 * enc_dim).to(coords.device)  # (1, 1, 1, 1, 2L)

    coords_time_freqs = coords.unsqueeze(-1) * freqs  # (B, N, S, 3, 1) * (1, 1, 1, 1, 2L) -> (B, N, S, 3, 2L)

    sin_part = torch.sin(coords_time_freqs[..., 0::2])
    cos_part = torch.cos(coords_time_freqs[..., 1::2])
    encoded_pos = torch.cat([sin_part.unsqueeze(-1), cos_part.unsqueeze(-1)], dim=-1)

    return encoded_pos.reshape((*coords_time_freqs.shape[:3], -1))


def positional_encoding_geom(coords, size, enc_dim):
    """Expand coordinates in the Fourier basis with geometrically spaced wavelengths from 2/D to 2pi"""
    freqs = torch.arange(enc_dim, dtype=torch.float, device=coords.device)
    freqs = (2 * np.pi * (size / 2) ** (freqs / (enc_dim - 1)))  # option 1: 2/D to 1
    freqs = freqs.view(*[1] * len(coords.shape), -1)  # 1 x 1 x D2
    coords = coords.unsqueeze(-1)  # B x 3 x 1
    x = torch.cat([torch.sin(coords * freqs), torch.cos(coords * freqs)], -1)  # B x 3 x D
    x = x.view(*coords.shape[:-2], enc_dim * 6)  # B x in_dim-zdim
    return x
