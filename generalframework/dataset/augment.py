from torchvision import transforms
import random
from PIL import ImageOps
import numpy as np, torch


def segment_transform(size):
    img_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor()
    ])
    mask_transform = transforms.Compose([
        transforms.Resize(size),
        ToLabel()
    ])
    return {'img': img_transform,
            'gt': mask_transform}


class ToLabel():
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, img):
        np_img = np.array(img)[None, ...]
        t_img = torch.from_numpy(np_img)
        return t_img.long()


def PILaugment(img_list):
    if random.random() > 0.5:
        img_list = [ImageOps.flip(img) for img in img_list]
    if random.random() > 0.5:
        img_list = [ImageOps.mirror(img) for img in img_list]
    if random.random() > 0.5:
        angle = random.random() * 90 - 45
        img_list = [img.rotate(angle) for img in img_list]

    if random.random() > 0.8:
        (w, h) = img_list[0].size
        crop = random.uniform(0.85, 0.95)
        W = int(crop * w)
        H = int(crop * h)
        start_x = w - W
        start_y = h - H
        x_pos = int(random.uniform(0, start_x))
        y_pos = int(random.uniform(0, start_y))
        img_list = [img.crop((x_pos, y_pos, x_pos + W, y_pos + H)) for img in img_list]

    return img_list
