import numpy as np
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from visdom import Visdom
import copy, os, shutil
import matplotlib.pyplot as plt
# plt.switch_backend('agg')
import torch
from pathlib import Path
from skimage import io, data


class Writter_tf(SummaryWriter):

    def __init__(self, writer_name: str, torchnet, num_img=20, random_seed=1, device='cpu') -> None:
        super().__init__(log_dir=writer_name)
        assert isinstance(num_img, int)
        self.writer_name = writer_name
        self.torchnet = torchnet
        self.random_seed = random_seed
        self.num_img = num_img
        self.device = device

    def cleanup(self, src='runs', des='archive'):
        self.export_scalars_to_json(os.path.join(self.writer_name, 'json.json'))
        self.close()
        # writerbasename = os.path.basename(self.writer_name)
        writerbasename = self.writer_name.replace('./runs/', '')
        shutil.move(os.path.join(src, writerbasename), os.path.join(des, writerbasename))

    def customized_add_image(self, img, gt, weak_gt, pred_mask, path, epoch):
        assert img.size(0) == 1

        fig = plt.figure()
        plt.imshow(img.data.cpu().squeeze().numpy(), cmap='gray')
        plt.contour(gt.data.cpu().squeeze().numpy(), levels=[0.5], colors="yellow", alpha=0.2, linewidth=0.001,
                    label='GT')
        plt.contour(weak_gt.data.cpu().squeeze().numpy(), levels=[0.5], colors="yellow", alpha=0.2, linewidth=0.001,
                    label='GT')
        pred = pred_mask
        [_, dice] = dice_loss(pred, gt)
        plt.contour(pred.data.cpu().squeeze().numpy(), levels=[0.5], level=[0],
                    colors="red", alpha=0.2, linewidth=0.001, label='CNN')
        plt.title('dice:%.3f' % dice)
        plt.axis('off')
        self.add_figure(path, fig, global_step=epoch)

    def add_images(self, dataloader, epoch, device='cpu', opt='tensorboard', omit_image=True):
        assert opt in ('tensorboard', 'save', 'both')
        dataset_name = dataloader.dataset.name
        np.random.seed(self.random_seed)
        dataset_ = copy.deepcopy(dataloader.dataset)
        dataset_.training = False
        if omit_image:
            np.random.seed(self.random_seed)
            selected_indxs = np.random.permutation(dataset_.imgs.__len__())[:self.num_img]
            selected_imgs = [dataset_.imgs[indx] for indx in selected_indxs]
            dataset_.imgs = selected_imgs
        from torch.utils.data import DataLoader
        dataloader_ = DataLoader(dataset_, batch_size=1)
        self.torchnet.eval()
        with torch.no_grad():
            for i, (img, gt, weak_gt, path) in enumerate(dataloader_):

                img, gt, weak_gt = img.to(device), gt.to(device), weak_gt.to(device)
                pred_mask = self.torchnet(img).max(1)[1]
                assert pred_mask.shape[0] == 1

                if opt == 'tensorboard' or opt == 'both':
                    self.customized_add_image(img, gt, weak_gt, pred_mask,
                                              os.path.join(dataset_name, os.path.basename(path[0])),
                                              epoch)
                if opt == 'save' or opt == 'both':
                    self.save_image(pred_mask, os.path.join(dataset_name, os.path.basename(path[0])), epoch)
        self.torchnet.train()

    def save_image(self, pred_mask, path, epoch):
        assert pred_mask.shape.__len__() == 3 and pred_mask.shape[0] == 1
        save_path = Path(self.writer_name) / str('%.3d' % epoch)
        save_path.mkdir(parents=True, exist_ok=True)
        save_path = save_path / str(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        pred_mask = pred_mask.data.cpu().numpy().squeeze()
        pred_mask[pred_mask == 1] = 255
        io.imsave(save_path, pred_mask)
