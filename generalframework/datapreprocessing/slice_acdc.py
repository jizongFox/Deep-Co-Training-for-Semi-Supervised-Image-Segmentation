#!/usr/bin/env python3.6

import re
import random
import argparse
import warnings
from pathlib import Path
from pprint import pprint
from functools import partial
from typing import Any, Callable, List, Tuple

import numpy as np
import nibabel as nib
from numpy import unique as uniq
from skimage.io import imsave
from skimage.transform import resize
import matplotlib.pyplot as plt

from .utils import mmap_, uc_, map_, augment


## normalized np.narray from [0,255], return dtype = np.uint8
def norm_arr(img: np.ndarray) -> np.ndarray:
    casted = img.astype(np.float32)
    shifted = casted - casted.min()
    norm = shifted / shifted.max()
    res = 255 * norm

    return res.astype(np.uint8)


##
def get_frame(filename: str, regex: str = ".*_frame(\d+)(_gt)?\.nii.*") -> str:
    matched = re.match(regex, filename)

    if matched:
        return matched.group(1)
    raise ValueError(regex, filename)


## get patient id
def get_p_id(path: Path) -> str:
    '''
    The patient ID, for the ACDC dataset, is the folder containing the data.
    '''
    res = path.parent.name

    assert "patient" in res, res
    # print res if assert is fault.
    return res


def save_slices(img_p: Path, gt_p: Path,
                dest_dir: Path, shape: Tuple[int], n_augment: int,
                img_dir: str = "img", gt_dir: str = "gt") -> Tuple[int, int, int, str, List[int]]:
    p_id: str = get_p_id(img_p)
    assert "patient" in p_id
    assert p_id == get_p_id(gt_p)

    f_id: str = get_frame(img_p.name)
    assert f_id == get_frame(gt_p.name)

    # Load the data
    img = np.asarray(nib.load(str(img_p)).dataobj)
    gt = np.asarray(nib.load(str(gt_p)).dataobj)

    assert img.shape == gt.shape
    assert img.dtype in [np.uint8, np.int16, np.float32]

    # Normalize and check data content
    norm_img = norm_arr(img)  # We need to normalize the whole 3d img, not 2d slices
    assert 0 == norm_img.min() and norm_img.max() == 255, (norm_img.min(), norm_img.max())
    assert gt.dtype == norm_img.dtype == np.uint8

    save_dir_img: Path = Path(dest_dir, img_dir)
    save_dir_gt: Path = Path(dest_dir, gt_dir)
    sizes_2d: np.ndarray = np.zeros(img.shape[-1])
    for j in range(img.shape[-1]):
        img_s = norm_img[:, :, j]
        gt_s = gt[:, :, j]
        assert img_s.shape == gt_s.shape

        # Resize and check the data are still what we expect
        resize_: Callable = partial(resize, mode="constant", preserve_range=True, anti_aliasing=False)
        r_img: np.ndarray = resize_(img_s, shape).astype(np.uint8)
        r_gt: np.ndarray = resize_(gt_s, shape).astype(np.uint8)
        assert r_img.dtype == r_gt.dtype == np.uint8
        assert 0 <= r_img.min() and r_img.max() <= 255  # The range might be smaller
        assert set(uniq(r_gt)).issubset(set(uniq(gt)))
        ## only calculate the gt ==3, ventrical surface.
        # sizes_2d[j] = r_gt[r_gt == 3].sum()
        sizes_2d[j] = (r_gt == 3).sum()

        for k in range(n_augment + 1):
            if k == 0:
                a_img, a_gt = r_img, r_gt
            else:
                ## the data augmentation is only with rotation, flip and mirror
                a_img, a_gt = map_(np.asarray, augment(r_img, r_gt))

            for save_dir, data in zip([save_dir_img, save_dir_gt], [a_img, a_gt]):
                filename = f"{p_id}_{f_id}_{k}_{j}.png"
                save_dir.mkdir(parents=True, exist_ok=True)

                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning)
                    imsave(str(Path(save_dir, filename)), data)
    ## return 3D size, minimal size for positive images. maximal size for positive image, f_id, and size_2d list
    return sizes_2d.sum(), sizes_2d[sizes_2d > 0].min(), sizes_2d.max(), f_id, sizes_2d


def main(args: argparse.Namespace):
    src_path: Path = Path(args.source_dir)
    dest_path: Path = Path(args.dest_dir)

    # Assume the cleaning up is done before calling the script
    assert src_path.exists()
    assert not dest_path.exists()

    # Get all the file names, avoid the temporal ones
    nii_paths: List[Path] = [p for p in src_path.rglob('*.nii.gz') if "_4d" not in str(p)]
    assert len(nii_paths) % 2 == 0, "Uneven number of .nii, one+ pair is broken"

    # We sort now, but also id matching is checked while iterating later on
    img_nii_paths: List[Path] = sorted(p for p in nii_paths if "_gt" not in str(p))
    gt_nii_paths: List[Path] = sorted(p for p in nii_paths if "_gt" in str(p))
    ## img_nii and gt_nii are paths for the images and ground truths respectively.
    assert len(img_nii_paths) == len(gt_nii_paths)
    paths: List[Tuple[Path, Path]] = list(zip(img_nii_paths, gt_nii_paths))

    print(f"Found {len(img_nii_paths)} pairs in total")
    pprint(paths[:5])

    validation_paths: List[Tuple[Path, Path]] = random.sample(paths, args.retain)
    training_paths: List[Tuple[Path, Path]] = [p for p in paths if p not in validation_paths]
    assert set(validation_paths).isdisjoint(set(training_paths))
    assert len(paths) == (len(validation_paths) + len(training_paths))

    for mode, _paths, n_augment in zip(["train", "val"], [training_paths, validation_paths], [args.n_augment, 0]):
        img_paths, gt_paths = zip(*_paths)  # type: Tuple[Any, Any]

        dest_dir = Path(dest_path, mode)
        print(f"Slicing {len(img_paths)} pairs to {dest_dir}")
        assert len(img_paths) == len(gt_paths)

        pfun = partial(save_slices, dest_dir=dest_dir, shape=args.shape, n_augment=n_augment)
        sizes = mmap_(uc_(pfun), zip(img_paths, gt_paths))
        # for paths in tqdm(list(zip(img_paths, gt_paths)), ncols=50):
        #     uc_(pfun)(paths)
        sizes_3d, sizes_2d_min, sizes_2d_max, frames, sizes_2d = map_(np.asarray, zip(*sizes))

        set1 = [e for s, f in zip(sizes_2d, frames) for e in s if f == '01' and e > 0]
        set2 = [e for s, f in zip(sizes_2d, frames) for e in s if f != '01' and e > 0]
        print(len(set1), len(set2))

        # plt.hist(set1, bins=50, alpha=0.5, label='01')
        # plt.hist(set2, bins=50, alpha=0.5, label='02')
        # plt.legend(loc='upper right')
        # plt.show()

        # for sizes in sizes_2d:
        #     trimmed = [e for e in sizes if e > 0]
        #     plt.hist(trimmed, bins=100, alpha=0.5)
        # plt.show()

        print("2d sizes: ", sizes_2d_min.min(), sizes_2d_max.max())
        print("3d sizes: ", sizes_3d.min(), sizes_3d.mean(), sizes_3d.max())


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Slicing parameters')
    parser.add_argument('--source_dir', type=str, required=True)
    parser.add_argument('--dest_dir', type=str, required=True)
    parser.add_argument('--img_dir', type=str, default="IMG")
    parser.add_argument('--gt_dir', type=str, default="GT")
    parser.add_argument('--shape', type=int, nargs="+", default=[256, 256])
    parser.add_argument('--retain', type=int, default=25, help="Number of retained patient for the validation data")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_augment', type=int, default=0,
                        help="Number of augmentation to create per image, only for the training set")
    args = parser.parse_args()
    random.seed(args.seed)

    print(args)

    return args


if __name__ == "__main__":
    args = get_args()
    random.seed(args.seed)

    main(args)
