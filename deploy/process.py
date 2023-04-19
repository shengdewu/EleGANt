import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
import faceutils as futils


class PreProcess:

    def __init__(self, img_size=256, landmark_npts=68, device='cpu'):
        self.img_size = img_size
        self.device = device

        xs, ys = np.meshgrid(
            np.linspace(
                0, self.img_size - 1,
                self.img_size
            ),
            np.linspace(
                0, self.img_size - 1,
                self.img_size
            )
        )
        xs = xs[None].repeat(landmark_npts, axis=0)
        ys = ys[None].repeat(landmark_npts, axis=0)
        fix = np.concatenate([ys, xs], axis=0)
        self.fix = torch.Tensor(fix)

        self.face_parse = futils.mask.FaceParser(device=device)

        self.up_ratio = 0.6 / 0.85
        self.down_ratio = 0.2 / 0.85
        self.width_ratio = 0.2 / 0.85
        self.lip_class = [7, 9]
        self.face_class = [1, 6]
        self.eyebrow_class = [2, 3]
        self.eye_class = [4, 5]

        self.transform = transforms.ToTensor()

    ############################## Mask Process ##############################
    # mask attribute: 0:background 1:face 2:left-eyebrow 3:right-eyebrow 4:left-eye 5: right-eye 6: nose
    # 7: upper-lip 8: teeth 9: under-lip 10:hair 11: left-ear 12: right-ear 13: neck
    def mask_process(self, mask: torch.Tensor):
        '''
        mask: (1, h, w)
        '''
        mask_lip = (mask == self.lip_class[0]).float() + (mask == self.lip_class[1]).float()
        mask_face = (mask == self.face_class[0]).float() + (mask == self.face_class[1]).float()

        # mask_eyebrow_left = (mask == self.eyebrow_class[0]).float()
        # mask_eyebrow_right = (mask == self.eyebrow_class[1]).float()
        mask_face += (mask == self.eyebrow_class[0]).float()
        mask_face += (mask == self.eyebrow_class[1]).float()

        mask_eye_left = (mask == self.eye_class[0]).float()
        mask_eye_right = (mask == self.eye_class[1]).float()

        # mask_list = [mask_lip, mask_face, mask_eyebrow_left, mask_eyebrow_right, mask_eye_left, mask_eye_right]
        mask_list = [mask_lip, mask_face, mask_eye_left, mask_eye_right]
        mask_aug = torch.cat(mask_list, 0)  # (C, H, W)
        return mask_aug

    def diff_process(self, lms: torch.Tensor, normalize=False):
        '''
        lms:(68, 2)
        '''

        assert lms.shape[0] == 68
        lms = lms.transpose(1, 0).reshape(-1, 1, 1)  # (136, 1, 1)

        diff = self.fix - lms  # (136, h, w)

        if normalize:
            norm = torch.norm(diff, dim=0, keepdim=True).repeat(diff.shape[0], 1, 1)
            norm = torch.where(norm == 0, torch.tensor(1e10), norm)
            diff /= norm
        return diff

    ############################## Compose Process ##############################
    def preprocess(self, image: np.ndarray, is_crop=True):
        '''
        return: image: np.ndarray, (H, W), mask: tensor, (1, H, W)
        '''
        face = futils.dlib.detect_np(image)
        # face: rectangles, List of rectangles of face region: [(left, top), (right, bottom)]
        if not face:
            return None, None, None

        face_on_image = face[0]

        # dis_img = image.copy()
        # cv2.rectangle(dis_img, (face_on_image.left(), face_on_image.top()), (face_on_image.right(), face_on_image.bottom()), color=(255, 0, 255), thickness=1)
        # cv2.imwrite('/mnt/sda1/valid.output/makeup/elgant-base4/demo/face.png', dis_img)

        if is_crop:
            image, face, crop_face = futils.dlib.crop_img_np(
                image, face_on_image, self.up_ratio, self.down_ratio, self.width_ratio)
        else:
            face = face[0]
            crop_face = None

        # dis_face = image.copy()
        # cv2.rectangle(dis_face, (face.left(), face.top()), (face.right(), face.bottom()), color=(255, 0, 255), thickness=1)
        # cv2.imwrite('/mnt/sda1/valid.output/makeup/elgant-base4/demo/face1.png', dis_face)

        # image: np.ndarray, cropped face
        # face: the same as above
        # crop face: rectangle, face region in cropped face
        mask = self.face_parse.parse(cv2.resize(image, (512, 512))).cpu()
        # obtain face parsing result
        # mask: Tensor, (512, 512)
        mask = F.interpolate(
            mask.view(1, 1, 512, 512),
            (self.img_size, self.img_size),
            mode="nearest").squeeze(0).long()  # (1, H, W)

        h, w, _ = image.shape
        assert h == w
        lms = futils.dlib.landmarks_np(image, face) * self.img_size / h  # scale to fit self.img_size
        # lms: narray, the position of 68 key points, (68 ,2)
        lms = torch.IntTensor(lms.round()).clamp_max_(self.img_size - 1)
        # distinguish upper and lower lips
        lms[61:64, 0] -= 1
        lms[65:68, 0] += 1
        for i in range(3):
            if torch.sum(torch.abs(lms[61 + i] - lms[67 - i])) == 0:
                lms[61 + i, 0] -= 1
                lms[67 - i, 0] += 1

        image = cv2.resize(image, dsize=(self.img_size, self.img_size), interpolation=cv2.INTER_AREA)

        # dis_img = image.copy()
        # for i in range(lms.shape[0]):
        #     y, x = lms[i].cpu().numpy().tolist()
        #     cv2.circle(dis_img, (x, y), 1, (255, 0, 0), 2)
        # cv2.imwrite('/mnt/sda1/valid.output/makeup/elgant-base4/demo/lms.png', dis_img)

        return [image, mask, lms], face_on_image, crop_face

    def process(self, image: np.ndarray, mask: torch.Tensor, lms: torch.Tensor):
        image = self.transform(image)
        mask = self.mask_process(mask)
        diff = self.diff_process(lms)
        return [image, mask, diff, lms]

    def __call__(self, image: np.ndarray, is_crop=True):
        source, face_on_image, crop_face = self.preprocess(image, is_crop)
        if source is None:
            return None, None, None
        return self.process(*source), face_on_image, crop_face

