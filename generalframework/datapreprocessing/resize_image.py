from PIL import Image
import numpy as np
from functools import reduce, partial
from multiprocessing import Pool
from pathlib import Path
from typing import Tuple
import os
import argparse

def meta_resize(img_path: Path, resolution=Tuple[int, int]):
    assert img_path.exists()
    img_name = img_path.name
    parent_folder_path = img_path.parent
    img: Image.Image = Image.open(img_path)
    new_img = img.resize(resolution, resample=Image.NEAREST if str(img_name).find('gtFine')>0 else Image.BICUBIC)
    new_parent_folder = Path(str(parent_folder_path).replace('Cityscapes','Cityscapes_new'))
    new_parent_folder.mkdir(parents=True,exist_ok=True)
    new_img.save(os.path.join(new_parent_folder, img_name))


def main(folder_path:str,resoltion:Tuple[int,int]):
    folder_path=Path(folder_path)
    assert folder_path.exists()
    img_paths = sorted(folder_path.glob('**/**/*.png'))
    resize_ = partial(meta_resize,resolution= resoltion)
    Pool().map(resize_,img_paths)


if __name__ == '__main__':
    main('/home/jizong/Workspace/ReproduceAdaptSegNet/data/Cityscapes/',resoltion=(1024,512))





