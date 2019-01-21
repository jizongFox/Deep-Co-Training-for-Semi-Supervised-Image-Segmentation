import numpy as np
import os
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt


def plot_one_csv_mod(file_path, plt, model='acc1', block=False):
    file = pd.read_csv(file_path)
    baseline = os.path.basename(file_path).replace('Namespace', '').replace('.csv', '')
    # plt.figure()
    plt.plot(file['acc1'], label=model.replace('acc', 'Model')+baseline)
    # plt.plot(file['acc2'], label="Model2_"+baseline)
    plt.legend()
    plt.grid()
    # plt.title(baseline)
    plt.ylim([0.3, 1])
    # print(baseline, 'best scores: acc1 {}, acc2 {}'.format(file['acc1'].max(), file['acc2'].max()))
    # plt.show(block=block)


def plot_dice_metrics_full(file_path, plot_title=''):
    file = pd.read_csv(file_path)
    f = plt.figure()
    file['train_dice'].plot()
    file['val_dice'].plot()
    file['val_batch_dice'].plot()
    plt.legend()
    plt.grid()
    plt.title(plot_title)
    plt.xlim([0.0, 100])
    plt.ylim([0.3, 1])
    return f


def plot_dice_metrics_partial(file_path, num_iter, plot_title=''):
    file = pd.read_csv(file_path)
    f = plt.figure()
    file['train_dice_'+str(num_iter)].plot()
    file['val_dice_'+str(num_iter)].plot()
    file['val_batch_dice_'+str(num_iter)].plot()
    plt.legend()
    plt.grid()
    plt.title(plot_title)
    plt.xlim([0.0, 100])
    plt.ylim([0.0, 1])
    return f


def plot_loss_metrics(file_path, plt, plot_title=''):
    file = pd.read_csv(file_path)
    f = plt.figure()
    file['train_loss'].plot()
    file['val_loss'].plot()
    plt.legend()
    plt.grid()
    plt.title(plot_title)
    plt.xlim([0.05, 100])
    plt.ylim([0.0, 0.5])
    return f


if __name__ == '__main__':
    # # =========================== Fully-supervised on Full dataset (My Experiment) ===========================
    # root = '/home/guillermo/Documents/InternshipETS/Results_Presentations/Jan_20_DCT_Seg/mine/archives/FS_fulldataset'
    # metrics_path = 'metrics.csv'
    # path_dice_file = os.path.join(root, metrics_path)
    # # plot dices
    # f1 = plot_dice_metrics_full(path_dice_file, plot_title='FS_fulldataset Dice Coefficient')
    # plt.show()
    # f1.savefig(os.path.join(root, 'dices_curves.png'))
    #
    # # plot losses
    # f2 = plot_loss_metrics(path_dice_file, plt, plot_title='FS_fulldataset Loss')
    # plt.show()
    # f2.savefig(os.path.join(root, 'losses_curves.png'))
    #
    # # =========================== Fully-supervised on Full dataset (Jizong's Experiment) ===========================
    # root = '/home/guillermo/Documents/InternshipETS/Results_Presentations/Jan_20_DCT_Seg/archives/cardiac/unet_No_agument/FS_fulldata'
    # metrics_path = 'metrics.csv'
    # path_dice_file = os.path.join(root, metrics_path)
    # # plot dices
    # f1 = plot_dice_metrics_full(path_dice_file, plot_title='FS_fulldata Dice Coefficient')
    # plt.show()
    # f1.savefig(os.path.join(root, 'dices_curves.png'))
    #
    # # plot losses
    # f2 = plot_loss_metrics(path_dice_file, plt, plot_title='FS_fulldata Loss')
    # plt.show()
    # f2.savefig(os.path.join(root, 'losses_curves.png'))

    # =========================== Fully-supervised on Partial dataset (Jizong's Experiment 1) ======================
    root = '/home/guillermo/Documents/InternshipETS/Results_Presentations/Jan_20_DCT_Seg/archives/cardiac/unet_No_agument/FS_partial'
    # # plot dices
    for metrics_path in ['metrics_0.csv', 'metrics_1.csv']:
        path_dice_file = os.path.join(root, metrics_path)
        for idx in range(3):
            f1 = plot_dice_metrics_partial(path_dice_file, idx+1, plot_title='FS_partial Dice Coefficient')
            plt.show()
            f1.savefig(os.path.join(root, metrics_path[:-4] + '_dices_curves_{}.png'.format(idx+1)))


    # # =========================== Fully-supervised on Partial dataset (Jizong's Experiment 2) ======================
    # root = '/home/guillermo/Documents/InternshipETS/Results_Presentations/Jan_20_DCT_Seg/archives/cardiac/unet_No_agument/FS_partial'
    # metrics_path = 'metrics_1.csv'
    # path_dice_file = os.path.join(root, metrics_path)
    # # plot dices
    # for idx in range(2):
    #     # f1 = plt.figure()
    #     f1 = plot_dice_metrics_partial(path_dice_file, idx+1, plot_title='FS_partial Dice Coefficient')
    #     plt.show()
    #     f1.savefig(os.path.join(root, metrics_path[:-4] + '_dices_curves_{}.png'.format(idx+1)))
    #
    # f1 = plot_dice_metrics_partial(path_dice_file, plot_title='FS_partial Dice Coefficient')
    # plt.show()
    # f1.savefig(os.path.join(root, metrics_path[:-4] + ' dices_curves.png'))
