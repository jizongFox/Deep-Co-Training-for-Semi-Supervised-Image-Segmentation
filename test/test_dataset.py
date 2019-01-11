from generalframework.dataset import MedicalImageDataset, PILaugment, segment_transform
import matplotlib.pyplot as plt
import time
from torch.utils.data import DataLoader


def test_dataset():
    dataroot: str = '../dataset/ACDC-all'
    train_set = MedicalImageDataset(dataroot, 'train', subfolders=['img', 'gt', 'gt2'],
                                    transform=segment_transform((256, 256)),
                                    augment=PILaugment, equalize=None, pin_memory=False,
                                    metainfo="[classSizeCalulator, {'C':4,'foldernames':['gt','gt2']}]"
                                    )
    train_loader = DataLoader(train_set, batch_size=1, num_workers=8)

    for i, (imgs, metainfo, filenames) in enumerate(train_loader):
        print(imgs)
        print(metainfo)
        print(filenames)
        time.sleep(2)



if __name__ == '__main__':
    test_dataset()
