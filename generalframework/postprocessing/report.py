import argparse
import operator
from pathlib import Path
from pprint import pprint

import pandas as pd


def get_args():
    parser = argparse.ArgumentParser(description='input folder path')
    parser.add_argument('--folder', type=str, required=True)
    parser.add_argument('--file', type=str, required=True)
    return parser.parse_args()


def main(args: argparse.Namespace):
    folder_path = Path(args.folder)
    assert folder_path.exists(), folder_path
    file_list = list(folder_path.glob('**/%s' % args.file))
    name_dict = {}
    for file in file_list:
        name_dict[file.parent.name] = str(file)
    pprint(name_dict)

    results = {}
    for k, v in name_dict.items():
        summary = pd.read_csv(v, index_col=0)
        mean_iou = summary['ensemble'].iloc[1:].mean()
        results[k] = summary['ensemble'].iloc[1:]
        results[k]['mean_iou'] = mean_iou
    results = pd.DataFrame(results)

    order_dict = results.loc['mean_iou'].to_dict()
    order_dict = dict(sorted(order_dict.items(), key=operator.itemgetter(1), reverse=True))
    results = results[list(order_dict.keys())]
    print('\nEnsemble score:\n',results)
    results.to_csv(folder_path / 'ensemble_results.csv')

    ## for average of the score
    results = {}
    for k, v in name_dict.items():
        summary = pd.read_csv(v, index_col=0)
        del summary['ensemble']
        results[k] = summary.mean(axis=1).iloc[1:]
        mean_iou = summary.mean(1).iloc[1:].mean()
        results[k]['mean_iou'] = mean_iou
    results = pd.DataFrame(results)

    order_dict = results.loc['mean_iou'].to_dict()
    order_dict = dict(sorted(order_dict.items(), key=operator.itemgetter(1), reverse=True))
    results = results[list(order_dict.keys())]
    print('\nAverage score:\n',results)
    results.to_csv(folder_path / 'mean_score_results.csv')


if __name__ == '__main__':
    main(get_args())
