from generalframework.dataset import MedicalImageDataset, PILaugment, segment_transform
import matplotlib.pyplot as plt
import time
from torch.utils.data import DataLoader
from generalframework.utils import iterator_
from pathlib import Path


def test_dataset():
    dataroot: str = '../dataset/ACDC-all'
    train_set = MedicalImageDataset(dataroot, 'train', subfolders=['img', 'gt'],
                                    transform=segment_transform((256, 256)),
                                    augment=PILaugment, equalize=None, pin_memory=False,
                                    metainfo="[classSizeCalulator, {'C':4,'foldernames':['gt','gt2']}]"
                                    )
    train_loader = DataLoader(train_set, batch_size=10, num_workers=8)

    n_batches = len(train_loader)

    for i, (imgs, metainfo, filenames) in enumerate(train_loader):
        print(imgs)
        print(metainfo)
        print(filenames)
        time.sleep(2)


def test_iter():
    dataroot: str = '../dataset/ACDC-all'
    train_set = MedicalImageDataset(dataroot, 'train', subfolders=['img', 'gt'],
                                    transform=segment_transform((256, 256)),
                                    augment=PILaugment, equalize=None, pin_memory=False,
                                    metainfo="[classSizeCalulator, {'C':4,'foldernames':['gt','gt2']}]"
                                    )
    train_loader_1 = DataLoader(train_set, batch_size=10, num_workers=8)
    train_loader_2 = DataLoader(train_set, batch_size=10, num_workers=8)

    n_batches_1 = len(train_loader_1)
    n_batches_2 = len(train_loader_2)
    assert n_batches_1 == n_batches_2

    train_loader_1_ = iterator_(train_loader_1)
    train_loader_2_ = iterator_(train_loader_2)

    output_list1 = []
    output_list2 = []

    for i in range(n_batches_2+1):
        data1 = train_loader_1_.__next__()
        data2 = train_loader_2_.__next__()
        output_list1.extend(data1[2])
        output_list2.extend(data2[2])
    assert set(output_list1) == set([Path(x).stem for x in train_loader_1.dataset.filenames['img']])
    assert set(output_list2) == set([Path(x).stem for x in train_loader_2.dataset.filenames['img']])


if __name__ == '__main__':
    test_iter()
