import functools
import itertools
import numbers
from copy import deepcopy as dc
from multiprocessing import Pool
from pathlib import Path
from pprint import pprint
from typing import Tuple

import fire
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from PIL import Image
from skimage.io import imsave
from torchvision.transforms import functional as F

plt.ion()


def arraynorm(array: np.ndarray):
    max, min = array.max(), array.min()
    array = (array - min) / (max - min)
    assert array.min() == 0 and array.max() == 1
    array = (array * 255.0).astype(np.uint8)
    assert array.max() == 255 and array.min() == 0
    return array


class Resize():
    '''
    Resize a serials of 2D image given pixeldim and targeted pixeldim, and max_dim
    '''

    def __init__(self,
                 pixeldim: Tuple[float, float],
                 t_pixeldim: Tuple[float, float] = (0.25, 0.25),
                 ) -> None:
        super().__init__()
        self.pixeldim: Tuple[float, float] = pixeldim
        self.t_pixeldim: Tuple[float, float] = t_pixeldim
        self.resize_ratio: Tuple[float, float] = tuple(np.array(self.pixeldim) / np.array(self.t_pixeldim))

    def __call__(self, img):
        r_img = Image.fromarray(img.astype(np.uint8)).resize(
            size=(int(img.shape[i] * self.resize_ratio[i]) for i in range(2)))
        return np.array(r_img)


class CenterCrop(object):
    """Crops the given PIL Image at the center.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped.

        Returns:
            PIL Image: Cropped image.
        """
        img = F.center_crop(img, self.size)
        return np.array(img)


def show_resolution(nii_path):
    nii_path = Path(nii_path)
    resol = nib.load(str(nii_path)).header.get_zooms()
    print(nib.load(str(nii_path)).shape)

    plt.imshow(nib.load(str(nii_path)).get_data()[:, :, 1])
    plt.show()
    plt.pause(0.5)
    return resol


def resize_nii(nii_path, o_path, resol=(0.25, 0.25)):
    nifile: nib.Nifti1Image = nib.load(nii_path)
    header: nib.Nifti1Header = nifile.header
    o_resol = header.get_zooms()[:2]
    new_header = dc(header)
    new_header['pixdim'][1:3] = resol

    origin_img: np.ndarray = nifile.get_data()
    f_resize = Resize(pixeldim=o_resol, t_pixeldim=resol)
    r_imgs = [f_resize(origin_img[:, :, i]) for i in range(origin_img.shape[2])]
    r_imgs = np.asarray(r_imgs, dtype=float).transpose((2, 1, 0))
    ni_img = nib.Nifti1Image(r_imgs, affine=None, header=new_header)
    assert np.allclose(ni_img.header.get_zooms()[:2], (0.25, 0.25))
    o_path: Path = Path(o_path)
    o_path.parent.mkdir(parents=True, exist_ok=True)
    print(f'{Path(nii_path).name}:resolution: {o_resol}->{resol}, with shape{nifile.shape}->{ni_img.shape}')
    nib.save(ni_img, str(o_path))


def save_slices(image_path: str, gt_path, save_dir: str, crop_size=(200, 200)):
    (Path(save_dir) / 'img').mkdir(parents=True, exist_ok=True)
    if gt_path:
        (Path(save_dir) / 'gt').mkdir(parents=True, exist_ok=True)
    img_nii: nib.Nifti1Image = nib.load(str(image_path))
    img = img_nii.get_data()

    if gt_path:
        gt_nii: nib.Nifti1Image = nib.load(str(gt_path))
        assert gt_nii.header.get_zooms() == img_nii.header.get_zooms()
        gt = gt_nii.get_data()

    resize = Resize(pixeldim=img_nii.header.get_zooms()[:2], t_pixeldim=(0.25, 0.25))

    img = arraynorm(img)

    for i in range(img.shape[-1]):
        s_img = img[:, :, i]
        if gt_path:
            s_gt = gt[:, :, i]
            assert s_img.shape == s_gt.shape

        r_img = resize(s_img)
        if gt_path:
            r_gt = resize(s_gt)

        c_img = CenterCrop(crop_size)(Image.fromarray(r_img))
        if gt_path:
            c_gt = CenterCrop(crop_size)(Image.fromarray(r_gt))

        imsave(Path(save_dir) / 'img' / (str(image_path.stem).replace('.nii','') + f'_{i}.png'), c_img)
        if gt_path:
            imsave(Path(save_dir) / 'gt' / (str(gt_path.stem).replace('.nii','') + f'_{i}.png'), c_gt)


def main(folder, save_dir):
    res_folder = Path(folder)
    assert res_folder.exists()
    assert (res_folder / 'train').exists()
    assert (res_folder / 'unlabel').exists()

    train_nii = list((res_folder / 'train').glob('*.nii.gz'))
    img_nii = sorted(p for p in train_nii if '-image' in str(p))
    gt_nii = sorted(p for p in train_nii if 'mask-r1' in str(p))
    assert img_nii.__len__() == gt_nii.__len__()
    print(f'Found {img_nii.__len__()} paired data')
    train_paths = list(zip(img_nii, gt_nii))
    pprint(train_paths[:3])
    unlabeled_paths = sorted((res_folder / 'unlabel').glob('*image.nii.gz'))
    print(f'Found {unlabeled_paths.__len__()} unlabeled data.')
    unlabeled_paths = list(zip(unlabeled_paths, itertools.repeat(None)))
    pprint(unlabeled_paths[:5])

    print('-> Began to perform slicing:')

    for paths, mode in zip([train_paths, unlabeled_paths], ['train', 'unlabeled']):
        save_slices_ = functools.partial(save_slices, save_dir='%s/%s' % (save_dir, mode))
        mmp(save_slices_, paths)
    print('-> Slicing ends with output at ./%s'%save_dir)

def mmp(func, iter_):
    P = Pool()
    P.starmap(func, iter_)


if __name__ == '__main__':
    fire.Fire()
