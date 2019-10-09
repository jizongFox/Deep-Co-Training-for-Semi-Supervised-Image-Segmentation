import json
import re
import warnings
from itertools import repeat
from multiprocessing import Pool
from pathlib import Path
from pprint import pprint
from typing import Union, Tuple

import nibabel
import numpy as np
from skimage.io import imsave
from skimage.transform import resize

_num = re.compile(r"\d+")
OUTOUT_SHAPE = (256, 256)


def _str2path(path: Union[str, Path]):
    assert isinstance(path, (str, Path)), path
    _path: Path = Path(path) if isinstance(path, str) else path
    return _path


def _normalize_array(array: np.ndarray) -> np.ndarray:
    array_: np.ndarray = (array - array.min()) / (array.max() - array.min())
    return array_.astype(np.float32)


def _read_nii(file_path: Union[str, Path], normalize=True) -> np.ndarray:
    """
    read
    :param file_path:
    :return:
    """
    _file_path = _str2path(file_path)
    assert _file_path.exists() and _file_path.is_file(), f"{_file_path} wrong."
    data = nibabel.load(str(_file_path)).get_data()
    norr_data = _normalize_array(data) if normalize else data
    norr_data = np.flip(norr_data.transpose((1, 0, 2)), axis=0)
    return norr_data


def read_patient_img_gt(img_path: str, gt_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    read image and ground truth data for the same patient
    :param img_path: image_path for one patient
    :param gt_path: gt_path for the same patient
    :return: np.ndarray of patient image and label array.
    """
    image_data = _read_nii(img_path, normalize=True)
    assert image_data.dtype == np.float32 and image_data.max() <= 1 and image_data.min() >= 0
    groundtruth_data = _read_nii(gt_path, normalize=False)
    assert set(np.unique(groundtruth_data)).issubset(set([0, 1]))
    return image_data, groundtruth_data


def slice_data(image_data: np.ndarray, gt_data: np.ndarray, patient_name: str, is_training=True) -> None:
    """
    save image and ground truth slices to disk
    :param image_data: 
    :param gt_data: 
    :param patient_name: 
    :return: 
    """
    main_folder = "train" if is_training else "val"

    Path(main_folder, "img").mkdir(exist_ok=True, parents=True)
    Path(main_folder, "gt").mkdir(exist_ok=True, parents=True)

    assert image_data.shape == gt_data.shape
    slice_num = image_data.shape[2]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        for i in range(slice_num):
            slice_image = resize(image_data[:, :, i], output_shape=OUTOUT_SHAPE, order=1,preserve_range=True)  #
            #         0: Nearest-neighbor
            #         1: Bi-linear (default)
            #         2: Bi-quadratic
            #         3: Bi-cubic
            #         4: Bi-quartic
            #         5: Bi-quintic
            slice_gt = resize(gt_data[:, :, i], output_shape=OUTOUT_SHAPE, order=0, preserve_range=True)
            imsave(f"{main_folder}/img/Patient_{patient_name}_{i:03d}.png", (slice_image * 255).astype(np.uint8))
            imsave(f"{main_folder}/gt/Patient_{patient_name}_{i:03d}.png", (slice_gt).astype(np.uint8))


def save_images(image_path: str, label_path: str, is_training: bool):
    assert set(map(lambda path: _num.search(path)[0], [image_path, label_path])).__len__() == 1
    p_num: str = "{:02d}".format(int(_num.search(image_path)[0]))
    image_data, gt_data = read_patient_img_gt(image_path, label_path)
    slice_data(image_data, gt_data, p_num, is_training)


def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--random_seed", type=int, default=0, help="random seed")
    parser.add_argument("--train_val_ratio", type=float, default=0.9, help="train_val_ratio (0.9)")
    args = parser.parse_args()
    pprint(args)
    return args


def main(args):
    train_validation_ratio = args.train_val_ratio
    np.random.seed(args.random_seed)
    with open("dataset.json", "r") as f:
        json_file = json.load(f)
    available_set = sorted(json_file["training"], key=lambda x: x["image"])
    permunate_set = np.random.permutation(available_set)
    train_set = permunate_set[:int(train_validation_ratio * len(available_set))]
    val_set = permunate_set[int(train_validation_ratio * len(available_set)):]
    assert len(train_set) + len(val_set) == len(permunate_set)
    print(f"training_set: {len(train_set)}")
    pprint(train_set[:5])
    print(f"val_set: {len(val_set)}")
    pprint(val_set[:5])
    pool = Pool()
    pool.starmap(save_images, zip([x["image"] for x in train_set], [x["label"] for x in train_set], repeat(True)))
    pool.starmap(save_images, zip([x["image"] for x in val_set], [x["label"] for x in val_set], repeat(False)))


if __name__ == "__main__":
    main(get_args())
