## todo write this report script to integrate all functions together.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import spline
from pathlib import Path
import os, sys
import argparse


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

    fig = plt.figure()
    for axis_, s in zip(args.axis, styles):
        plt.clf()

        for filepath, c in zip(filepaths, colors):
            folername_ = filepath.parent.name
            # metrics_file = pd.read_csv(filepath)
            metrics_file = np.load(filepath)

            for seg in range(metrics_file.shape[2]):
                x = np.arange(0, metrics_file.shape[0])
                # x = np.array(metrics_file.index.tolist())

                # value = metrics_file.iloc[:, axis_].tolist()
                value = metrics_file[:, :, seg, axis_].mean(1)
                value_name = f'DSC{axis_}'
                if args.interpolate:
                    new_x = np.linspace(0, len(x) - 1, (len(x) - 1) * 8)
                    new_value = spline(x, np.array(value), new_x, 3)
                    # assert new_value.min() == np.array(value).min()
                    # assert new_value.max() == np.array(value).max()
                    x, y = new_x, new_value
                else:
                    x, y = x, value
                label = f'Seg_{seg}' + value_name
                plt.plot(x, y, label=label + ' ' + folername_, linestyle=s)
                print(f'{filepath}, {label}: {np.array(value).max()}')

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
    args = choices.parse_args()
    print(args)
    return args


if __name__ == '__main__':
    main(get_args())
