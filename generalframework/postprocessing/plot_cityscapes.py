## todo write this report script to integrate all functions together.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import spline
from pathlib import Path
import os, sys
import argparse
import matplotlib

matplotlib.use('Agg')


def main(args: argparse.Namespace) -> None:
    colors = ["c", "r", "g", "b", "m", 'y', 'k', 'chartreuse', 'coral']
    styles = [':', '--', '-.', ]
    assert args.folders.__len__() <= colors.__len__()
    # assert args.axis.__len__() <= styles.__len__()
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
    for style in styles:
        plt.clf()

        for filepath, c in zip(filepaths, colors):
            foldername_ = filepath.parent.name
            # metrics_file = pd.read_csv(filepath)
            metrics_file = np.nanmean(np.load(filepath)[:,args.num_seg],1)
            metrics = filepath.stem.split('_')[2]

            value_name = build_mean_plot(plt, metrics, metrics_file, filepath, foldername_, args.interpolate, style)

        if not args.draw_all:
            plt.title(file + ' ' + value_name)
            plt.grid()
            plt.legend()
            plt.tight_layout()
            if args.y_lim != [0, 0]:
                plt.ylim(args.y_lim)

            if args.show:
                plt.show()
            fig.savefig(str(filepath.parents[1]) + '/' + args.postfix + '_' + filepath.stem + '.png')

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


def build_mean_plot(plt, metrics, metrics_file, filepath, foldername_, interpolate, style, quite=False):
    x = np.arange(0, metrics_file.shape[0])
    value = metrics_file
    value_name = f'{metrics}'
    if interpolate:
        new_x = np.linspace(0, len(x) - 1, (len(x) - 1) * 8)
        new_value = spline(x, np.array(value), new_x, 3)
        x, y = new_x, new_value
    else:
        x, y = x, value
    label = f'Seg_MV_' + value_name
    plt.plot(x, y, label=label + ' ' + foldername_, linestyle=style)
    if not quite:
        print(f'{filepath}, {label}: {np.array(value).max()}')
    return value_name


def get_args() -> argparse.Namespace:
    choices = argparse.ArgumentParser(description='input folders and files to report the training')
    choices.add_argument('--folders', type=str, nargs='+', help='folders that contain the reported file', required=True)
    choices.add_argument('--file', type=str, help='input the filename to report', required=True)
    choices.add_argument('--interpolate', action='store_true')
    # choices.add_argument('--axis', type=int, nargs='+', help='the axis number choice to draw', required=True)
    choices.add_argument('--num_seg',type=int, default=0)
    choices.add_argument('--show', action='store_true')
    choices.add_argument('--y_lim', type=float, nargs='*', help='[y_min, y_max]', default=[0, 0])
    choices.add_argument('--draw_all', action='store_true', help='draw all together')
    choices.add_argument('--postfix', type=str, default='', required=True)
    args = choices.parse_args()
    print(args)
    return args


if __name__ == '__main__':
    main(get_args())
