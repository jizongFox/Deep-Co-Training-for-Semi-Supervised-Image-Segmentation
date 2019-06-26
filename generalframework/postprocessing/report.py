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
    if set([x.parent.name for x in file_list]).__len__() == len(file_list):
        for file in file_list:
            name_dict[file.parent.name] = str(file)
    else:
        for file in file_list:
            name_dict[file] = str(file)

    pprint(name_dict)

    # load div results:
    div = {}
    for k, v in name_dict.items():
        div_csv = pd.read_csv(v.replace(args.file, 'div.csv'))
        try:
            div[k] = div_csv.mean(1).values[0]
        except:
            import ipdb
            ipdb.set_trace()
    kappa = pd.DataFrame(div, index=['kappa'])

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
    results = results.append(kappa, sort=False)
    print('\nEnsemble score:\n', results)

    results.T.to_csv(folder_path / 'ensemble_results.csv')

    # for average of the score
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
    try:
        results = results.append(kappa, sort=False)
    except:
        pass
    print('\nAverage score:\n', results)
    results.T.to_csv(folder_path / 'mean_score_results.csv')


if __name__ == '__main__':
    main(get_args())
