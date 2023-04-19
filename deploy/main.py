import torch
from models.model import Generator
from .process import PreProcess
import numpy as np
import cv2
from torchvision.utils import save_image


def norm(x: torch.Tensor):
    return x * 2 - 1


def de_norm(x: torch.Tensor):
    out = (x + 1) / 2
    return out.clamp(0, 1)


def get_generator(model_path):
    parmas = {'conv_dim': 64,
              'image_size': 256,
              'num_head': 1,
              'double_encoder': False,
              'use_ff': False,
              'num_layer_e': 3,
              'num_layer_d': 2,
              'window_size': 16,
              'merge_mode': 'conv'}
    model = Generator(**parmas)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    return model


class Main:
    def __init__(self, model_path, device='cuda'):
        self.process = PreProcess(device=device)
        self.device = device
        self.model = get_generator(model_path).to(self.device)
        return

    def prepare_input(self, *data_inputs):
        """
        data_inputs: List[image, mask, diff, lms]
        """
        inputs = []
        for i in range(len(data_inputs)):
            if i == 0:
                inputs.append(norm(data_inputs[i]).to(self.device).unsqueeze(0))
            else:
                inputs.append(data_inputs[i].to(self.device).unsqueeze(0))
        # prepare mask
        inputs[1] = torch.cat((inputs[1][:, 0:1], inputs[1][:, 1:].sum(dim=1, keepdim=True)), dim=1)

        # mask = inputs[1]
        # assert mask.shape[0] == 1 and mask.shape[1] == 2
        # save_image(torch.cat([mask[0][0].unsqueeze(0), mask[0][1].unsqueeze(0)], dim=-1), '/mnt/sda1/valid.output/makeup/elgant-base4/demo/mask.png', normalize=False)
        return inputs

    def generate(self, image_A, mask_A, diff_A, lms_A, image_B, mask_B, diff_B, lms_B):
        """image_A is content, image_B is style"""
        with torch.no_grad():
            res = self.model(image_A, image_B, mask_A, mask_B, diff_A, diff_B, lms_A, lms_B)
        return res

    def __call__(self, imgA: np.ndarray, imgB: np.ndarray):
        imgA = cv2.cvtColor(imgA, cv2.COLOR_BGR2RGB)
        imgB = cv2.cvtColor(imgB, cv2.COLOR_BGR2RGB)

        source_input, face, crop_face = self.process(imgA)
        reference_input, _, _ = self.process(imgB)
        if not (source_input and reference_input):
            return None, None

        source_input = self.prepare_input(*source_input)
        reference_input = self.prepare_input(*reference_input)
        fakeA = self.generate(*source_input, *reference_input)
        fakeA = de_norm(fakeA).squeeze(0)

        fakeA = cv2.cvtColor(fakeA.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy(), cv2.COLOR_RGB2BGR)

        return fakeA, crop_face
