# -*- coding: utf-8 -*-
import torch
import os
from .image_pipe.compose import Compose
from .post.box_bridge import BoxPost
import numpy as np


class FaceDetection:

    def __init__(self, device='cpu'):

        checkout_point_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'checkpoint')
        self.face_model = torch.jit.load(os.path.join(checkout_point_path, 'face_box.pt')).to(device)
        self.post = BoxPost()
        self.transformer = Compose()
        self.device = device
        return

    def __call__(self, input_img: np.ndarray, confidence=0.5):
        """
        Args :
            input_img: 输入图片为BGR
            confidence
        Return
                "faces": [  # 返回值
                     {"x1": 0, "y1": 0, "x2": 0, "y2": 0},  # 脸部矩形框；
                ]
        """
        data_info = dict()
        data_info['is_load'] = False
        data_info['img'] = input_img

        data = self.transformer(data_info)

        with torch.no_grad():
            feats = self.face_model(data['img'].to(self.device).unsqueeze(0))
            bboxes = self.post.simple_infer(feats, [data['img_metas']])
            assert len(bboxes) == 1  # batch size
            assert len(bboxes[0]) == 1  # class size
            bboxes = bboxes[0][0]

        face = list()
        for i in range(bboxes.shape[0]):
            bbox = bboxes[i].tolist()[:4]
            score = float(bboxes[i][4])
            if score <= confidence:
                continue
            rect = [int(p + 0.5) for p in bbox]
            face_rect = {'x1': rect[0], 'y1': rect[1], 'x2': rect[2], 'y2': rect[3]}
            face.append(face_rect)

        return face



