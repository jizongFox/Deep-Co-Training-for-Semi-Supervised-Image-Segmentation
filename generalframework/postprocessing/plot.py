## todo write this report script to integrate all functions together.
import argparse
import os
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.interpolate import spline

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main(args: argparse.Namespace) -> None:
    colors = ["c", "r", "g", "b", "m", 'y', 'k', 'chartreuse', 'coral']
    styles = [':', '--', '-.', ]
    assert args.folders.__len__() <= colors.__len__()
    assert args.axis.__len__() <= styles.__len__()
    assert args.y_lim.__len__() == 2
    assert args.y_lim[0] <= args.y_lim[1]

    folders: list = args.folders
    assert folders.__len__() > 0, f'folder list length:{folders.__len__()}'
    file: str = args.file
    filepaths: list[Path] = [Path(os.path.join(folder, file)) for folder in folders]
    for filepath in filepaths:
        assert filepath.exists(), f'{filepath.name} doesn\'t exisit'

    value_name = ''
    fig = plt.figure()
    for axis_, s in zip(args.axis, styles):
        plt.clf()

        for filepath, c in zip(filepaths, colors):
            folername_ = filepath.parent.name
            metrics_file = np.load(filepath)

            segs = range(metrics_file.shape[2])
            if args.seg_id in segs:
                segs = [args.seg_id]
                value_name = build_plot(plt, axis_, segs, metrics_file, filepath, folername_,
                                        args.interpolate, s)
            else:
                value_name = build_plot(plt, axis_, segs, metrics_file, filepath, folername_,
                                        args.interpolate, s)

        if not args.draw_all:
            plt.title(file + ' ' + value_name)
            plt.grid()
            plt.legend()
            plt.tight_layout()
            if args.y_lim != [0, 0]:
                plt.ylim(args.y_lim)

            if args.show:
                plt.show()
            fig.savefig(str(filepath.parents[1]) + '/' + args.postfix + '_' + filepath.stem + "_" + value_name + '.png')

    if args.draw_all:
        plt.title(file + ' ' + 'all')

        plt.grid()
        plt.legend()
        plt.tight_layout()
        if args.y_lim != [0, 0]:
            plt.ylim(args.y_lim)

        if args.show:
            plt.show()
        fig.savefig(str(filepath.parents[1]) + '/' + args.postfix + '_' + 'all.png')


def build_plot(plt, axis_, seg_lst, metrics_file, filepath, folername_, interpolate, style, quite=False):
    for seg in seg_lst:
        x = np.arange(0, metrics_file.shape[0])
        value = metrics_file[:, seg, axis_,0 ]
        value_name = f'DSC{axis_}'
        if interpolate:
            new_x = np.linspace(0, len(x) - 1, (len(x) - 1) * 8)
            new_value = spline(x, np.array(value), new_x, 3)
            x, y = new_x, new_value
        else:
            x, y = x, value
        label = f'Seg_{seg}_' + value_name
        plt.plot(x, y, label=label + ' ' + folername_, linestyle=style)
        if not quite:
            print(f'{filepath}, {label}: {np.array(value).max()}')
    return value_name


def get_args() -> argparse.Namespace:
    choices = argparse.ArgumentParser(description='input folders and files to report the training')
    choices.add_argument('--folders', type=str, nargs='+', help='folders that contain the reported file', required=True)
    choices.add_argument('--file', type=str, help='input the filename to report', required=True)
    choices.add_argument('--interpolate', action='store_true')
    choices.add_argument('--axis', type=int, nargs='+', help='the axis number choice to draw', required=True)
    choices.add_argument('--show', action='store_true')
    choices.add_argument('--y_lim', type=float, nargs='*', help='[y_min, y_max]', default=[0, 0])
    choices.add_argument('--draw_all', action='store_true', help='draw all together')
    choices.add_argument('--postfix', type=str, default='', required=True)
    choices.add_argument('--seg_id', type=int, default=None,
                         help='Optional index of the segmentator to be plotted, otherwise all segmentators are reported')
    args = choices.parse_args()
    print(args)
    return args


if __name__ == '__main__':
    main(get_args())
