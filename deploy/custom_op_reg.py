import torch
from torch.onnx import register_custom_op_symbolic
import os


def register_custom_op():
    def linalg_pinv(g, input):
        return g.op("custom_domain::linalg_pinv", input)
    register_custom_op_symbolic("custom::linalg_pinv", linalg_pinv, 9)

    def grid_sampler(g, input, grid, interpolation_mode, padding_mode, align_corners):
        return g.op("custom_domain::grid_sampler", input, grid, interpolation_mode, padding_mode, align_corners)
    register_custom_op_symbolic("custom::grid_sampler", grid_sampler, 9)


path = os.path.dirname(__file__)

torch.ops.load_library(f'{path}/custom_ops.cpython-36m-x86_64-linux-gnu.so')

register_custom_op()

inverse = torch.ops.custom.linalg_pinv
grid_sampler = torch.ops.custom.grid_sampler


