import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules.module_base import ResidualBlock_IN, Downsample, Upsample, PositionalEmbedding, MergeBlock
from .modules.module_attn import Attention_apply, FeedForwardLayer, MultiheadAttention
from .modules.sow_attention import SowAttention
from .modules.tps_transform import tps_spatial_transform
from .modules.pseudo_gt import expand_area
from .modules.pseudo_gt import mask_blend


class Generator(nn.ModuleDict):
    """Generator. Encoder-Decoder Architecture."""

    def __init__(self, conv_dim=64, image_size=256, num_layer_e=2, num_layer_d=1, window_size=16, use_ff=False,
                 merge_mode='conv', num_head=1, double_encoder=False, lms_points=68, **unused):
        super(Generator, self).__init__()

        self.img_size = image_size
        # -------------------------- Encoder --------------------------

        layers = nn.Conv2d(3, conv_dim, kernel_size=7, stride=1, padding=3, bias=False)
        self.add_module('in_conv', layers)

        # Down-Sampling & Bottleneck
        curr_dim = conv_dim
        feature_size = image_size
        for i in range(2):
            layers = Downsample(curr_dim, curr_dim * 2, affine=True)
            self.add_module('down_{:d}'.format(i + 1), layers)
            curr_dim = curr_dim * 2
            feature_size = feature_size // 2

            self.add_module('e_bottleneck_{:d}'.format(i + 1),
                            nn.Sequential(*[ResidualBlock_IN(curr_dim, curr_dim, affine=True) for j in range(num_layer_e)])
                            )

        ### second encoder
        self.double_encoder = double_encoder
        if self.double_encoder:
            layers = nn.Conv2d(3, conv_dim, kernel_size=7, stride=1, padding=3, bias=False)
            self.add_module('in_conv_s', layers)

            # Down-Sampling & Bottleneck
            curr_dim = conv_dim
            feature_size = image_size
            for i in range(2):
                layers = Downsample(curr_dim, curr_dim * 2, affine=True)
                self.add_module('down_{:d}_s'.format(i + 1), layers)
                curr_dim = curr_dim * 2
                feature_size = feature_size // 2

                self.add_module('e_bottleneck_{:d}_s'.format(i + 1),
                                nn.Sequential(*[ResidualBlock_IN(curr_dim, curr_dim, affine=True) for j in range(num_layer_e)])
                                )

        # --------------------------- Transfer ----------------------------
        curr_dim = conv_dim
        feature_size = image_size
        self.use_ff = use_ff
        for i in range(2):
            curr_dim = curr_dim * 2
            feature_size = feature_size // 2
            self.add_module('embedding_{:d}'.format(i + 1), PositionalEmbedding(
                embedding_dim=136,
                feature_size=feature_size,
                max_size=image_size,
                embedding_type='l2_norm'
            ))
            if i < 1:
                self.add_module('attention_extract_{:d}'.format(i + 1), SowAttention(
                    window_size=window_size,
                    in_channels=curr_dim + 136,
                    proj_channels=curr_dim + 136,
                    value_channels=curr_dim,
                    out_channels=curr_dim,
                    num_heads=num_head
                ))
            else:
                self.add_module('attention_extract_{:d}'.format(i + 1), MultiheadAttention(
                    in_channels=curr_dim + 136,
                    proj_channels=curr_dim + 136,
                    value_channels=curr_dim,
                    out_channels=curr_dim,
                    num_heads=num_head
                ))

            if use_ff:
                self.add_module('feedforward_{:d}'.format(i + 1), FeedForwardLayer(curr_dim, curr_dim))
            self.add_module('attention_apply_{:d}'.format(i + 1), Attention_apply(curr_dim))

            # --------------------------- Decoder ----------------------------

        # Bottleneck & Up-Sampling & Merge
        for i in range(2):
            self.add_module('d_bottleneck_{:d}'.format(i + 1),
                            nn.Sequential(*[ResidualBlock_IN(curr_dim, curr_dim, affine=True) for j in range(num_layer_d)])
                            )
            layers = Upsample(curr_dim, curr_dim // 2, affine=True)
            self.add_module('up_{:d}'.format(i + 1), layers)
            curr_dim = curr_dim // 2
            if i < 1:
                self.add_module('merge_{:d}'.format(i + 1), MergeBlock(merge_mode, curr_dim))

        layers = nn.Sequential(
            nn.InstanceNorm2d(curr_dim, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False),
        )
        self.add_module('out_conv', layers)

        self.fix_grid = self._create_grid(image_size, image_size, lms_points)
        self.image_size = image_size
        return

    def get_transfer_input(self, image, mask, lms, is_reference=False):
        diff = self.fix_grid.to(image.device) - lms.transpose(2, 1).reshape(1, -1, 1, 1)
        feature_size = image.shape[2]
        scale_factor = 1.0
        fea_list, mask_list, diff_list, lms_list = [], [], [], []

        # input conv
        if self.double_encoder and is_reference:
            fea = self['in_conv_s'](image)
        else:
            fea = self['in_conv'](image)

        # down-sampling & bottleneck
        for i in range(2):
            if self.double_encoder and is_reference:
                fea = self['down_{:d}_s'.format(i + 1)](fea)
                fea_ = self['e_bottleneck_{:d}_s'.format(i + 1)](fea)
            else:
                fea = self['down_{:d}'.format(i + 1)](fea)
                fea_ = self['e_bottleneck_{:d}'.format(i + 1)](fea)
            fea_list.append(fea_)

            feature_size = feature_size // 2
            scale_factor = scale_factor * 0.5
            mask_ = F.interpolate(mask, feature_size, mode='nearest')
            mask_list.append(mask_)

            diff_ = self['embedding_{:d}'.format(i + 1)](diff, mask)
            diff_list.append(diff_)

            lms_ = lms * scale_factor
            lms_list.append(lms_)

        return [fea_list, mask_list, diff_list, lms_list]

    def get_transfer_output(self, fea_c_list, mask_c_list, diff_c_list, lms_c_list,
                            fea_s_list, mask_s_list, diff_s_list, lms_s_list):
        attn_out_list = []
        for i in range(2):
            feature_size = fea_c_list[i].shape[2]

            # align
            if i == 0:
                fea_s_ = self.tps_align(feature_size, lms_s_list[i], lms_c_list[i], fea_s_list[i])
                mask_s_ = self.tps_align(feature_size, lms_s_list[i], lms_c_list[i], mask_s_list[i], 'nearest')
                diff_s_ = self.tps_align(feature_size, lms_s_list[i], lms_c_list[i], diff_s_list[i], 'nearest')
            else:
                fea_s_ = fea_s_list[i]
                mask_s_ = mask_s_list[i]
                diff_s_ = diff_s_list[i]

            # transfer
            input_q = torch.cat((fea_c_list[i], diff_c_list[i]), dim=1)
            input_k = torch.cat((fea_s_, diff_s_), dim=1)
            attn_out = self['attention_extract_{:d}'.format(i + 1)](input_q, input_k, fea_s_, mask_c_list[i], mask_s_)
            if self.use_ff:
                attn_out = self['feedforward_{:d}'.format(i + 1)](attn_out)
            attn_out_list.append(attn_out)

        return attn_out_list

    def decode(self, fea_c_list, attn_out_list):
        # apply
        for i in range(2):
            fea_c_ = self['attention_apply_{:d}'.format(i + 1)](fea_c_list[i], attn_out_list[i])
            fea_c_ = self['d_bottleneck_{:d}'.format(2 - i)](fea_c_)
            fea_c_list[i] = fea_c_

        # up-sampling & merge
        fea_c = fea_c_list[1]
        for i in range(2):
            fea_c = self['up_{:d}'.format(i + 1)](fea_c)
            if i < 1:
                fea_c = self['merge_{:d}'.format(i + 1)](fea_c_list[0], fea_c)

        fea_c = self['out_conv'](fea_c)
        return fea_c

    def forward(self, c, s, mask_c, mask_s, lms_c, lms_s):
        """
        c: content, stands for source image. shape: (b, c, h, w)
        s: style, stands for reference image. shape: (b, c, h, w)
        mask_c: (b, c', h, w)
        diff: (b, d, h, w)
        lms: (b, K, 2)
        """
        diff_c = self.fix_grid - lms_c.transpose(2, 1).reshape(1, -1, 1, 1)
        diff_s = self.fix_grid - lms_s.transpose(2, 1).reshape(1, -1, 1, 1)

        transfer_input_c = self.get_transfer_input(c, mask_c, diff_c, lms_c)
        transfer_input_s = self.get_transfer_input(s, mask_s, diff_s, lms_s, True)
        attn_out_list = self.get_transfer_output(*transfer_input_c, *transfer_input_s)
        fea_c = self.decode(transfer_input_c[0], attn_out_list)
        return fea_c

    def tps_align(self, feature_size, lms_s, lms_c, fea_s, sample_mode='bilinear'):
        '''
        fea: (B, C, H, W), lms: (B, K, 2)
        '''
        fea_out = []
        for l_s, l_c, f_s in zip(lms_s, lms_c, fea_s):
            l_c = torch.flip(l_c, dims=[1]) / (feature_size - 1)
            l_s = (torch.flip(l_s, dims=[1]) / (feature_size - 1)).unsqueeze(0)
            f_s = f_s.unsqueeze(0)  # (1, C, H, W)
            fea_trans, _ = tps_spatial_transform(feature_size, feature_size, l_c, f_s, l_s, sample_mode)
            fea_out.append(fea_trans)
        return torch.cat(fea_out, dim=0)

    def generate_partial_mask(self, source_mask, mask_area='full', saturation=1.0, eyeblur=None):
        """
        source_mask: (C, H, W), lip, face, left eye, right eye
        return: apply_mask: (1, H, W)
        """
        if eyeblur is None:
            eyeblur = {'margin': 12, 'blur_size': 7}

        assert mask_area in ['full', 'skin', 'lip', 'eye']
        if mask_area == 'full':
            return torch.sum(source_mask[0:2], dim=0, keepdim=True) * saturation
        elif mask_area == 'lip':
            return source_mask[0:1] * saturation
        elif mask_area == 'skin':
            mask_l_eye = expand_area(source_mask[2:3], eyeblur['margin'])  # * source_mask[1:2]
            mask_r_eye = expand_area(source_mask[3:4], eyeblur['margin'])  # * source_mask[1:2]
            mask_eye = mask_l_eye + mask_r_eye
            # mask_eye = mask_blend(mask_eye, 1.0, source_mask[1:2], blur_size=self.eyeblur['blur_size'])
            mask_eye = mask_blend(mask_eye, 1.0, blur_size=eyeblur['blur_size'])
            return source_mask[1:2] * (1 - mask_eye) * saturation
        elif mask_area == 'eye':
            mask_l_eye = expand_area(source_mask[2:3], eyeblur['margin'])  # * source_mask[1:2]
            mask_r_eye = expand_area(source_mask[3:4], eyeblur['margin'])  # * source_mask[1:2]
            mask_eye = mask_l_eye + mask_r_eye
            # mask_eye = mask_blend(mask_eye, saturation, source_mask[1:2], blur_size=self.eyeblur['blur_size'])
            mask_eye = mask_blend(mask_eye, saturation, blur_size=eyeblur['blur_size'])
            return mask_eye

    def generate_reference_sample(self, apply_mask=None, source_mask=None, mask_area=None, saturation=1.0):
        """
        all the operations on the mask, e.g., partial mask, saturation,
        should be finally defined in apply_mask
        """
        if source_mask is not None and mask_area is not None:
            apply_mask = self.generate_partial_mask(source_mask, mask_area, saturation)
            apply_mask = apply_mask.unsqueeze(0).to(source_mask.device)

        if apply_mask is None:
            apply_mask = torch.ones(1, 1, self.img_size, self.img_size).to(source_mask.device)
        return apply_mask

    # def forward(self, image, mask, lms, t_image, t_mask, t_lms, apply_mask):
    #     """
    #     Input: a source sample and multiple reference samples
    #     Return: PIL.Image, the fused result
    #     """
    #     device = image[0].device
    #
    #     # encode source
    #     source_sample_transfer_input = self.get_transfer_input(image, mask, lms)
    #
    #     # encode references
    #     reference_sample_transfer_input = self.get_transfer_input(t_image, t_mask, t_lms, is_reference=True)
    #
    #     # self attention
    #     source_sample_attn_out_list = self.get_transfer_output(
    #         *source_sample_transfer_input, *source_sample_transfer_input
    #     )
    #
    #     # full transfer for each reference
    #     reference_sample_attn_out_list = self.get_transfer_output(
    #         *source_sample_transfer_input, *reference_sample_transfer_input
    #     )
    #
    #     # fusion
    #     # if the apply_mask is changed without changing source and references,
    #     # only the following steps are required
    #     fused_attn_out_list = []
    #     for i in range(len(source_sample_attn_out_list)):
    #         init_attn_out = torch.zeros_like(source_sample_attn_out_list[i], device=device)
    #         fused_attn_out_list.append(init_attn_out)
    #     apply_mask_sum = torch.zeros((1, 1, self.img_size, self.img_size), device=device)
    #
    #     apply_mask_sum += apply_mask
    #     for i in range(len(source_sample_attn_out_list)):
    #         feature_size = reference_sample_attn_out_list[i].shape[2]
    #         apply_mask = F.interpolate(apply_mask, feature_size, mode='nearest')
    #         fused_attn_out_list[i] += apply_mask * reference_sample_attn_out_list[i]
    #
    #     # self as reference
    #     source_apply_mask = 1 - apply_mask_sum.clamp(0, 1)
    #     for i in range(len(source_sample_attn_out_list)):
    #         feature_size = source_sample_attn_out_list[i].shape[2]
    #         apply_mask = F.interpolate(source_apply_mask, feature_size, mode='nearest')
    #         fused_attn_out_list[i] += apply_mask * source_sample_attn_out_list[i]
    #
    #     # decode
    #     result = self.decode(
    #         source_sample_transfer_input[0], fused_attn_out_list
    #     )
    #     return result

    def _create_grid(self, w, h, landmark_points):
        ys, xs = torch.meshgrid(
            torch.linspace(
                0, w - 1,
                w
            ),
            torch.linspace(
                0, h - 1,
                h
            )
        )
        xs = xs[None].repeat(landmark_points, 1, 1)
        ys = ys[None].repeat(landmark_points, 1, 1)
        fix = torch.cat([ys, xs], dim=0)
        return fix.unsqueeze(0)