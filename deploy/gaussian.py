import torch
import torch.nn.functional as tnf
import math


def get_gaussian_kernel(kernel_size=3, sigma=0.8, in_channels=3, out_channels=3):
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1) / 2
    variance = sigma ** 2

    gaussian_kernel = (1./(2 * math.pi * variance)) * torch.exp(-torch.sum((xy_grid - mean) ** 2, dim=-1) / (2 * variance))

    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(in_channels, 1, 1, 1)
    return gaussian_kernel


def gaussian(img, kernel_size=3, sigma=2):
    b, in_channel, h, w = img.shape
    gaussian_kernel = get_gaussian_kernel(kernel_size, sigma, in_channel).to(img.device)
    return tnf.conv2d(img, gaussian_kernel, padding=kernel_size // 2, groups=in_channel, stride=1)
