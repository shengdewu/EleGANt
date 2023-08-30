import torch
from models.model import Generator
from models.elegant_onnx import Generator as GeneratorOnnx
from .process import PreProcess
import numpy as np
import cv2
import onnx
import onnxruntime
import os
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
        self.model.eval()
        return

    def prepare_input(self, *data_inputs, no_face=False):
        """
        data_inputs: List[image, mask, lms]
        """
        inputs = []
        for i in range(len(data_inputs)):
            if i == 0:
                inputs.append(norm(data_inputs[i]).to(self.device).unsqueeze(0))
            else:
                inputs.append(data_inputs[i].to(self.device).unsqueeze(0))

        # prepare mask
        # lip_mask = inputs[1][:, 0:1, :]

        if no_face:
            inputs[1] = torch.cat((inputs[1][:, 0:1], inputs[1][:, 2:].sum(dim=1, keepdim=True)), dim=1)
        else:
            inputs[1] = torch.cat((inputs[1][:, 0:1], inputs[1][:, 1:].sum(dim=1, keepdim=True)), dim=1)

        mask = inputs[1]
        assert mask.shape[0] == 1 and mask.shape[1] == 2
        return inputs

    def generate(self, image_A, mask_A, lms_A, image_B, mask_B, lms_B):
        """image_A is content, image_B is style"""
        with torch.no_grad():
            res = self.model(image_A, image_B, mask_A, mask_B, lms_A, lms_B)
        return res

    def post(self, fake):
        fakeA = de_norm(fake).squeeze(0)
        fakeA = cv2.cvtColor(fakeA.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy(), cv2.COLOR_RGB2BGR)
        return fakeA

    def __call__(self, imgA: np.ndarray, imgB: np.ndarray):
        imgA = cv2.cvtColor(imgA, cv2.COLOR_BGR2RGB)
        imgB = cv2.cvtColor(imgB, cv2.COLOR_BGR2RGB)

        source_input, face, crop_face = self.process(imgA)
        reference_input, _, _ = self.process(imgB)
        if not (source_input and reference_input):
            return None, None

        source_input = self.prepare_input(*source_input, no_face=False)
        reference_input = self.prepare_input(*reference_input, no_face=False)
        fakeA = self.generate(*source_input, *reference_input)

        # mask = gaussian(face_mask, kernel_size=5)
        # save_image(torch.cat([face_mask, mask], dim=-1), '/mnt/sda1/valid.output/makeup/elgant-eye-lip/mask.png', normalize=False)
        # fakeMA = mask * fakeA + (1-mask) * source_input[0]

        return self.post(fakeA), crop_face

    def partial_transfer(self, imgA: np.ndarray, imgB: np.ndarray, mask_area='skin', saturation=1.0):
        assert mask_area in ['lip', 'skin', 'eye']

        imgA = cv2.cvtColor(imgA, cv2.COLOR_BGR2RGB)
        imgB = cv2.cvtColor(imgB, cv2.COLOR_BGR2RGB)

        source_input, face, crop_face = self.process(imgA)
        reference_input, _, _ = self.process(imgB)

        if not (source_input and reference_input):
            return None, None

        source_mask = source_input[1]
        ref_mask = reference_input[1]

        source_sample = self.prepare_input(*source_input, no_face=False)
        reference_sample = self.prepare_input(*reference_input, no_face=False)

        reference_apply_mask = self.model.generate_reference_sample(source_mask=source_mask, mask_area=mask_area, saturation=saturation)

        save_image(torch.cat([source_mask[0].unsqueeze(0), source_mask[1].unsqueeze(0), source_mask[2].unsqueeze(0), source_mask[3].unsqueeze(0)], dim=-1), '/mnt/sda1/valid.output/makeup/elgant-noface/mask.png', normalize=False)

        onnx_name = '/mnt/sda1/workspace/open_source/EleGANt/deploy/elegant.onnx'

        torch.onnx.export(self.model,
                          (source_sample[0],
                           source_sample[1],
                           source_sample[2],
                           reference_sample[0],
                           reference_sample[1],
                           reference_sample[2],
                           reference_apply_mask,
                           ),
                          onnx_name,
                          export_params=False,
                          input_names=['sample_img', 'sample_mask', 'sample_pts',
                                       'target_img', 'target_mask', 'target_pts'
                                                                    'apply_mask'],
                          output_names=['fake'],
                          opset_version=14)

        model = onnx.load(onnx_name)
        onnx.checker.check_model(model)
        print(onnx.helper.printable_graph(model.graph))

        # ort_session = onnxruntime.InferenceSession(onnx_name,
        #                                            providers=['CPUExecutionProvider',
        #                                                       'CUDAExecutionProvider'])
        # input_dict = {'sample_img': source_sample[0].cpu().numpy(),
        #               'sample_mask': source_sample[1].cpu().numpy(),
        #               'sample_pts': source_sample[2].cpu().numpy(),
        #               'target_img': reference_sample[0].cpu().numpy(),
        #               'target_mask': reference_sample[1].cpu().numpy(),
        #               'target_pts': reference_sample[2].cpu().numpy(),
        #               'apply_mask': reference_apply_mask.cpu().numpy()}
        #
        # outputs = ort_session.run(None, input_dict)
        # fakeA = torch.from_numpy(outputs[0])

        fakeA = self.model.partial_forward(source_sample[0],
                                           source_sample[1],
                                           source_sample[2],
                                           reference_sample[0],
                                           reference_sample[1],
                                           reference_sample[2], reference_apply_mask.to(self.device))
        return self.post(fakeA), crop_face
