from generalframework import LOGGER, flags, app, config_logger
from generalframework.dataset import MedicalImageDataset, segment_transform, augment, get_dataset_root
from generalframework.loss import get_loss_fn
from generalframework.arch import get_arch
from generalframework.trainer import ADMM_Trainer
from generalframework.utils import extract_from_big_dict
import torch
import warnings

warnings.filterwarnings('ignore')
torch.set_num_threads(1)

def build_datasets(hparams):
    root_dir = get_dataset_root(hparams['dataroot'])
    train_dataset = MedicalImageDataset(root_dir, 'train', transform=segment_transform((256, 256)),
                                        augment=augment if hparams['data_aug'] else None)
    val_dataset = MedicalImageDataset(root_dir, 'val', transform=segment_transform((256, 256)), augment=None)

    return train_dataset, val_dataset

def check_consistance(hparams):
    if hparams['method']=='fullysupervised':
        assert hparams['loss']=='cross_entropy'
        assert hparams['num_admm_innerloop']==1
    else:
        assert hparams['loss']=='partial_ce'
        assert hparams['num_admm_innerloop']>1,hparams['num_admm_innerloop']
        assert hparams['batch_size']==1,hparams['batch_size']

def run(argv):
    del argv

    hparams = flags.FLAGS.flag_values_dict()
    check_consistance(hparams)
    train_dataset, val_dataset = build_datasets(hparams)

    arch_hparams = extract_from_big_dict(hparams, AdmmGCSize.arch_hparam_keys)
    torchnet = get_arch(arch_hparams['arch'], arch_hparams)

    admm = get_method(hparams['method'], torchnet, **hparams)
    criterion = get_loss_fn(hparams['loss'])
    trainer = ADMM_Trainer(admm, [train_dataset, val_dataset], criterion, hparams)
    trainer.start_training()


if __name__ == '__main__':
    torch.manual_seed(41)
    flags.DEFINE_string('dataroot', default='cardiac', help='the name of the dataset')
    flags.DEFINE_boolean('data_aug', default=False, help='data_augmentation')
    flags.DEFINE_string('loss',default='partial_ce',help='loss used in admm loop')
    # AdmmSize.setup_arch_flags()
    AdmmGCSize.setup_arch_flags()
    # ADMM_size_inequality.setup_arch_flags()
    # ADMM_reg_size_inequality.setup_arch_flags()
    ADMM_Trainer.setup_arch_flags()
    app.run(run)
