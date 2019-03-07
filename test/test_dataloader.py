from generalframework.dataset import MedicalImageDataset, segment_transform, augment
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import numpy as np


def show_img_mask_weakmask(img, gt, weak):
    from matplotlib.colors import ListedColormap

    # Random data
    data1 = np.random.random((4, 4))

    # Choose colormap
    cmap = pl.cm.Greens

    # Get the colormap colors
    my_cmap = cmap(np.arange(cmap.N))

    # Set alpha
    my_cmap[:, -1] = np.linspace(0, 1, cmap.N)
    my_cmap[:128, -1] = 0

    # Create new colormap
    my_cmap = ListedColormap(my_cmap)

    gt = gt.squeeze()
    fig = plt.figure(1)
    y_min, y_max = np.argwhere(gt.sum(0) > 0)[0][0], np.argwhere(gt.sum(0) > 0)[-1][0]
    x_min, x_max = np.argwhere(gt.sum(1) > 0)[0][0], np.argwhere(gt.sum(1) > 0)[-1][0]
    margin = 150
    plt.clf()
    plt.imshow(img.squeeze()[x_min - margin:x_max + margin, y_min - margin:y_max + margin], cmap='gray')
    plt.contourf(gt.squeeze()[x_min - margin:x_max + margin, y_min - margin:y_max + margin], colors='red',
                 levels=[0.5, 0.8])

    y = np.linspace(0, x_max - x_min + 2 * margin, x_max - x_min + 2 * margin)
    x = np.linspace(0, y_max - y_min + 2 * margin, y_max - y_min + 2 * margin)
    X, Y = np.meshgrid(x, y)
    plt.contourf(X, Y, weak.squeeze()[x_min - margin:x_max + margin, y_min - margin:y_max + margin], 4, cmap=my_cmap,
                 levels=1)
    plt.tight_layout()
    plt.axis('off')
    plt.show()
    plt.pause(0.1)


def test_dataloader():
    root_dir = '../dataset/ACDC-2D-All'
    train_dataset = MedicalImageDataset(root_dir, 'train', transform=segment_transform((500, 500)), augment=augment)
    val_dataset = MedicalImageDataset(root_dir, 'val', transform=segment_transform((500, 500)), augment=None)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    for i, (Img, GT, wgt, _) in enumerate(train_loader):
        if GT.sum() <= 0 or wgt.sum() <= 0:
            continue
        show_img_mask_weakmask(Img.numpy(), GT.numpy(), wgt.numpy())

        # # ToPILImage()(Img[0]).show()
        # if i == 5:
        #     train_loader.dataset.set_mode('eval')
        # ToPILImage()(Img[0]).show()
        # if i == 10:
        #     break

    # for i, (img, gt, wgt, _) in enumerate(val_loader):
    #     ToPILImage()(img[0]).show()
    #     if i == 5:
    #         val_loader.dataset.set_mode('eval')
    #     ToPILImage()(img[0]).show()
    #     if i == 10:
    #         break
    assert train_dataset.__len__() == train_dataset.imgs.__len__()


def test_prostate_dataloader():
    root_dir = '../dataset/PROSTATE'
    train_dataset = MedicalImageDataset(root_dir, 'train', transform=segment_transform((128, 128)), augment=augment)
    val_dataset = MedicalImageDataset(root_dir, 'val', transform=segment_transform((128, 128)), augment=None)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # for i, (Img, GT, wgt, _) in enumerate(train_loader):
    #     ToPILImage()(Img[0]).show()
    #     if i == 5:
    #         train_loader.dataset.set_mode('eval')
    #     ToPILImage()(Img[0]).show()
    #     if i == 10:
    #         break
    #
    # for i, (img, gt, wgt, _) in enumerate(val_loader):
    #     ToPILImage()(img[0]).show()
    #     if i == 5:
    #         val_loader.dataset.set_mode('eval')
    #     ToPILImage()(img[0]).show()
    #     if i == 10:
    #         break
    assert train_dataset.__len__() == train_dataset.imgs.__len__()


if __name__ == '__main__':
    test_dataloader()
