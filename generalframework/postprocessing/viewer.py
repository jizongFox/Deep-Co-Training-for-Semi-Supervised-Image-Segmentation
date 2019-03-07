#!/usr/bin/env python3.6

import re
import argparse
from pathlib import Path
from pprint import pprint
from functools import partial
from typing import Callable, Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from skimage.io import imread
from skimage.transform import resize
from matplotlib.widgets import Button


## math the pattern and the string
def extract(pattern: str, string: str) -> str:
    try:
        return re.match(pattern, string).group(1)
    except AttributeError:  # id not found
        return None


def display_item(axe, img: np.ndarray, mask: np.ndarray, contour: bool):
    m = resize(mask, img.shape, mode='constant', preserve_range=True)
    try:
        assert (img.shape == m.shape)
    except AssertionError:
        # print(title)
        # print(img.shape, m.shape)
        # raise

        # Some grayscale mask are sometimes loaded with 3 channel
        m = m[:, :, 0]

    axe.imshow(img, cmap="gray")

    if contour:
        axe.contour(m, cmap='rainbow')
    else:
        axe.imshow(m, alpha=.5)
    axe.axis('off')


def display(background_names: List[str], segmentation_names: List[List[str]],
            indexes: List[int], column_title: List[str], row_title: List[str],
            crop: int, contour: bool, remap: Dict, fig=None) -> None:
    if not fig:
        fig = plt.figure()
    gs = gridspec.GridSpec(len(indexes),  # the different image length
                           len(segmentation_names))  # the folder name

    for i, idx in enumerate(indexes):
        img: np.ndarray = imread(background_names[idx])

        if crop > 0:
            img = img[crop:-crop, crop:-crop]

        for j, names in enumerate(segmentation_names):
            axe = fig.add_subplot(gs[i, j])

            seg: np.ndarray = imread(names[idx])
            if crop > 0:
                seg = seg[crop:-crop, crop:-crop]
            if remap:
                for k, v in remap.items():
                    seg[seg == k] = v

            display_item(axe, img, seg, contour)

            if j == 0:
                print(row_title[idx])
                axe.text(-30, seg.shape[1] // 2, row_title[idx], rotation=90,
                         verticalalignment='center', fontsize=14)
            if i == 0:
                axe.set_title(column_title[j])

    # fig.show()
    # plt.show()
    # plt.draw()
    fig.show()


def get_image_lists(img_source: str, folders: List[str], id_regex: str) -> Tuple[List[str], List[List[str]], List[str]]:
    path_source: Path = Path(img_source)
    background_names: List[str] = sorted(map(str, path_source.glob("*")))
    segmentation_names: List[List[str]] = [sorted(map(str, Path(folder).glob("*"))) for folder in folders]

    extracter: Callable[[str], str] = partial(extract, id_regex)
    background_names = [bg for bg in background_names if extracter(bg) is not None]
    segmentation_names = [[sn for sn in sl if extracter(sn) is not None] for sl in segmentation_names]

    ids: List[str] = list(map(extracter, background_names))

    for names, folder in zip(segmentation_names, folders):
        try:
            assert (len(background_names) == len(names))
            assert (ids == list(map(extracter, names)))
        except AssertionError:
            print(f"Error verifying content for folder {folder}")
            print(f"Background folder '{img_source}': {len(background_names)} imgs")
            pprint(background_names[:10])
            print(f"Folder '{folder}': {len(names)} imgs")
            pprint(names[:10])

    return background_names, segmentation_names, ids


class EventHandler(object):
    def __init__(self, order: List[int], n: int, draw_function: Callable, fig):
        self.order: List[int] = order
        self.draw_function: Callable = draw_function
        self.n = n
        self.i = 0
        self.fig = fig

    def __call__(self, event):
        print("pouet", event)

        if event.button == 1:  # next
            self.i += 1
        elif event.button == 3:  # prev
            self.i -= 1

        a = self.i * self.n

        self.redraw(a)

    def redraw(self, a):
        self.fig.clear()

        idx: List[int] = self.order[a:a + self.n]

        self.draw_function(idx, fig=self.fig)

        self.fig.canvas.draw()


def main() -> None:
    args: argparse.Namespace = get_args()
    np.random.seed(args.seed)

    background_names: List[str]
    segmentation_names: List[List[str]]
    ids: List[str]
    background_names, segmentation_names, ids = get_image_lists(args.img_source, args.folders, args.id_regex)

    if args.display_names is None:
        display_names = [f for f in args.folders]
    else:
        assert len(args.display_names) == len(args.folders), (args.display_names, args.folders)
        display_names = args.display_names

    order: List[int] = list(range(len(background_names)))
    order = np.random.permutation(order)

    draw_function = partial(display, background_names, segmentation_names,
                            column_title=display_names,
                            row_title=ids,
                            crop=args.crop,
                            contour=not args.no_contour,
                            remap=eval(args.remap))

    fig = plt.figure()
    event_handler = EventHandler(order, args.n, draw_function, fig)
    fig.canvas.mpl_connect('button_press_event', event_handler)

    draw_function(order[:args.n], fig=fig)
    plt.show()

    # for a in range(0, len(background_names), args.n):
    #     idx: List[int] = order[a:a + args.n]
    #     assert(len(idx == args.n))
    #     draw_function(idx)
    #     plt.show()


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Display the requested data.")
    parser.add_argument("-n", type=int, default=4,
                        help="The number of images to sample per window.")
    parser.add_argument("--img_source", type=str, required=True,
                        help="The folder containing the images (background).")
    parser.add_argument("--seed", type=int, default=0,
                        help="The seed for the number generator. Used to sample the images. \
                             Useful to reproduce the same outputs between runs.")
    parser.add_argument("--id_regex", type=str, default=".*/(.*).png",
                        help="The regex to extract the image id from the images names \
                             Required to match the images between them.")
    parser.add_argument("folders", type=str, nargs='*',
                        help="The folder containing the source segmentations.")
    parser.add_argument("--display_names", type=str, nargs='*',
                        help="The display name for the folders in the viewer")
    parser.add_argument("--crop", type=int, default=0,
                        help="The number of pixels to remove from each border")
    parser.add_argument("--no_contour", action="store_true",
                        help="Do not draw a contour but a transparent overlap instead.")
    parser.add_argument("--remap", type=str, default="{}",
                        help="Remap some mask values if needed. Useful to suppress some classes.")
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    main()
