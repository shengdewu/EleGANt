import onnx
import onnxruntime
import torch
import numpy as np
import os
import cv2


class LandmarkDetect:
    def __init__(self):
        self.onnx_name = 'cv/pfld.onnx'
        base_name = os.path.basename(__file__)
        onnx_path = os.path.abspath(__file__).replace(base_name, self.onnx_name)
        model = onnx.load(onnx_path)
        onnx.checker.check_model(model)

        self.ort_session = onnxruntime.InferenceSession(onnx_path,
                                                        providers=['CPUExecutionProvider', 'CUDAExecutionProvider'])

        node_arg = self.ort_session.get_inputs()
        assert len(node_arg) == 1
        node_arg = node_arg[0]
        self.name = node_arg.name
        self.shape = node_arg.shape
        self.type = node_arg.type
        return

    def __call__(self, img, box, is_padding=False):
        # fx, fy, fx + fw, fy + fh
        x1, y1, x2, y2 = box
        face = img[y1: y2, x1: x2, :]
        h, w, c = face.shape
        if is_padding:
            scale = max(h, w) / self.shape[-1]
            nw, nh = int(h / scale + 0.5), int(w / scale + 0.5)

            face = cv2.resize(face, dsize=(nw, nh))
            left = 0
            right = 0
            top = 0
            bottom = 0
            if nh != self.shape[-1]:
                offset = self.shape[-1] - nh
                top = offset // 2
                bottom = offset - top
            elif nw != self.shape[-1]:
                offset = self.shape[-1] - nw
                left = offset // 2
                right = offset - left

            if left != 0 or right != 0 or top != 0 or bottom != 0:
                face = cv2.copyMakeBorder(face, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
        else:
            face = cv2.resize(face, dsize=(self.shape[-1], self.shape[-2]))

        face = (face / 255.0).astype(np.float32)

        outputs = self.ort_session.run(None, {self.name: face.transpose((2, 0, 1))[np.newaxis, :]})

        output = (outputs[0].reshape(-1, 2) * [w, h] + [x1 + 0.5, y1 + 0.5]).astype(np.int)
        return output

