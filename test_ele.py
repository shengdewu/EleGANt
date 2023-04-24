import os
import argparse
import numpy as np
import cv2
import torch

from training.config import get_config
from training.utils import create_logger, print_args

from deploy.main import Main
from color_transform.reinhard import color_transfer


def recover(tarImg, fakeImg):
    h, w, _ = fakeImg.shape
    assert h == w
    height, width = tarImg.shape[:2]
    small_source = cv2.resize(tarImg, (w, h))
    laplacian_diff = tarImg.astype(np.float) - cv2.resize(small_source, (width, height)).astype(np.float)
    fakeA = (cv2.resize(fakeImg, (width, height)) + laplacian_diff).round().clip(0, 255)
    fakeA = fakeA.astype(np.uint8)
    # fakeA = cv2.fastNlMeansDenoisingColored(fakeA)
    return fakeA


def main(config, args):

    logger = create_logger(args.save_folder, args.name, 'info', console=True)
    print_args(args, logger)
    logger.info(config)

    inference = Main(args.load_path, 'cuda')

    makeup_pair = list()
    with open('makeup_pair_compare.txt', mode='r') as f:
        names = f.readlines()
        for name in names:
            name = name.strip('\n')
            makeup_pair.append(name.split(' '))

    for i, (imga_name, imgb_name) in enumerate(makeup_pair):

        out_name = f'{args.save_folder}/{imga_name.split("/")[-1]}-{imgb_name.split("/")[-1]}.jpg'

        makeup_file = os.path.join(args.reference_dir, imgb_name)
        nonmakeup_file = os.path.join(args.source_dir, imga_name)
        if not os.path.exists(makeup_file) or not os.path.exists(nonmakeup_file):
            continue

        imgA = cv2.imread(nonmakeup_file, cv2.IMREAD_COLOR)
        imgB = cv2.imread(makeup_file, cv2.IMREAD_COLOR)

        fake, crop_face = inference(imgA, imgB)

        if fake is None:
            continue

        imgA = imgA[crop_face.top():crop_face.bottom(), crop_face.left():crop_face.right(), :]

        fakeA = recover(imgA, fake)

        fakeA1 = color_transfer(imgA, fakeA, preserve_paper=False)
        ratio = 0.5
        fakeA2 = (fakeA.astype(np.float) * ratio + fakeA1.astype(np.float) * (1 - ratio)).astype(np.uint8)
        ratio = 0.7
        fakeA3 = (fakeA1.astype(np.float) * ratio + imgA.astype(np.float) * (1 - ratio)).astype(np.uint8)

        h, w, _ = imgA.shape
        imgB = cv2.resize(imgB, dsize=(w, h), interpolation=cv2.INTER_AREA)

        vis_image = np.hstack((imgA, fakeA, fakeA1, fakeA2, fakeA3, imgB))
        cv2.imwrite(out_name, vis_image)


def demo(config, args):

    logger = create_logger(args.save_folder, args.name, 'info', console=True)
    print_args(args, logger)
    logger.info(config)

    inference = Main(args.load_path, 'cuda')

    n_imgname = sorted(os.listdir(args.source_dir))
    m_imgname = sorted(os.listdir(args.reference_dir))

    for i, (imga_name, imgb_name) in enumerate(zip(n_imgname, m_imgname)):

        out_name = f'{args.save_folder}/{imga_name.split("/")[-1]}-{imgb_name.split("/")[-1]}.jpg'

        makeup_file = os.path.join(args.reference_dir, imgb_name)
        nonmakeup_file = os.path.join(args.source_dir, imga_name)
        if not os.path.exists(makeup_file) or not os.path.exists(nonmakeup_file):
            continue

        imgA = cv2.imread(nonmakeup_file, cv2.IMREAD_COLOR)
        imgB = cv2.imread(makeup_file, cv2.IMREAD_COLOR)

        fake, crop_face = inference.partial_transfer(imgA, imgB, 'lip')

        if fake is None:
            continue

        imgA = imgA[crop_face.top():crop_face.bottom(), crop_face.left():crop_face.right(), :]

        fakeA = recover(imgA, fake)

        h, w, _ = imgA.shape
        imgB = cv2.resize(imgB, dsize=(w, h), interpolation=cv2.INTER_AREA)

        vis_image = np.hstack((imgA, imgB, fakeA))
        cv2.imwrite(out_name, vis_image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("argument for training")
    parser.add_argument("--name", type=str, default='demo')
    parser.add_argument("--save_path", type=str, default='/mnt/sda1/valid.output/makeup/elgant-test', help="path to save model")
    parser.add_argument("--load_path", type=str, help="folder to load model",
                        default='/mnt/sda1/train.output/makeup.output/elegant/elegant/epoch_50/G.pth')

    # parser.add_argument("--source-dir", type=str, default="/mnt/sda2/makeup.data/MT-xintu/images")
    # parser.add_argument("--reference-dir", type=str, default="/mnt/sda2/makeup.data/MT-xintu/images")
    parser.add_argument("--source-dir", type=str, default="/mnt/sda1/workspace/open_source/EleGANt/assets/images/non-makeup")
    parser.add_argument("--reference-dir", type=str, default="/mnt/sda1/workspace/open_source/EleGANt/assets/images/makeup")
    parser.add_argument("--gpu", default='0', type=str, help="GPU id to use.")

    args = parser.parse_args()
    args.gpu = 'cuda:' + args.gpu
    args.device = torch.device(args.gpu)

    args.save_folder = os.path.join(args.save_path, args.name)
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)

    config = get_config()
    demo(config, args)