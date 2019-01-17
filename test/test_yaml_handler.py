from generalframework.utils.utils import YAMLConfig
from absl import flags, app


def get_default_parameter():
    # TODO: Provide a proper documentation for each parameter
    # arch parameters
    flags.DEFINE_string('arch.name', default='enet', help='architecture to be trained')
    flags.DEFINE_integer('arch.num_classes', default=4, help='number of classes')
    # optim parameters
    flags.DEFINE_string('optim.name', default='enet', help='name of the optimizer')
    flags.DEFINE_float('optim.lr', default=0.01, help='initial learning rate')
    # scheduler parameters
    flags.DEFINE_string('scheduler.name', default='enet', help='name of the optimizer')
    flags.DEFINE_multi_integer('scheduler.milestones', default=[10,20,30,40,50,60,70,80,90],
                               help='milestones used for scheduler')
    flags.DEFINE_float('scheduler.gamma', default=0.5, help='gamma for lr_scheduler')
    # dataset parameters
    flags.DEFINE_string('dataset.root_dir', default='enet', help='root folder of the dataset')
    flags.DEFINE_multi_string('dataset.subfolders', default=['img', 'gt'],
                              help='subfolder corresponding to images and groundtruths')

    flags.DEFINE_string('dataset.transform', default='segment_transform((256, 256))',
                        help='transformation for imgs and gts')
    flags.DEFINE_string('dataset.augment', default='PILaugment', help='data augmentation function')

    flags.DEFINE_boolean('dataset.pin_memory', default=True, help='')
    # lab_dataloader parameters
    flags.DEFINE_boolean('lab_dataloader.pin_memory', default=True, help='')
    flags.DEFINE_integer('lab_dataloader.batch_size', default=2, help='rnumber of batch size')
    flags.DEFINE_integer('lab_dataloader.num_workers', default=2, help='number of workers used in dataloader')
    flags.DEFINE_boolean('lab_dataloader.shuffle', default=True, help='')
    flags.DEFINE_boolean('lab_dataloader.drop_last', default=True, help='')
    # unlab_dataloader parameters
    flags.DEFINE_boolean('unlab_dataloader.pin_memory', default=False, help='')
    flags.DEFINE_integer('unlab_dataloader.batch_size', default=2, help='rnumber of batch size')
    flags.DEFINE_integer('unlab_dataloader.num_workers', default=2, help='number of workers used in dataloader')
    flags.DEFINE_boolean('unlab_dataloader.shuffle', default=True, help='')
    flags.DEFINE_boolean('unlab_dataloader.drop_last', default=True, help='')
    # trainer parameters
    flags.DEFINE_integer('trainer.max_epoch', default=100, help='')
    flags.DEFINE_string('trainer.save_dir', default='tmp/cotraining', help='')
    flags.DEFINE_string('trainer.device', default='cuda', help='')
    flags.DEFINE_multi_integer('scheduler.axises', default=[0,1,2,3], help='')
    flags.DEFINE_string('trainer.metricname', default='test.csv', help='')
    flags.DEFINE_integer('trainer.lambda_cot_max', default=10, help='')
    flags.DEFINE_float('trainer.lambda_adv_max', default=0.5, help='gamma for lr_scheduler')
    flags.DEFINE_integer('trainer.epoch_max_ramp', default=80, help='')
    flags.DEFINE_integer('trainer.ramp_up_mult', default=-5, help='')
    # loss parameters
    flags.DEFINE_string('loss.name', default='cross_entropy', help='architecture to be trained')
    flags.DEFINE_multi_float('loss.weight', default=[0.01, 1., 1., 1.], help='number of classes')
    # start_training parameters
    flags.DEFINE_boolean('start_training.train_jsd', default=True, help='')
    flags.DEFINE_boolean('start_training.train_adv', default=True, help='')
    flags.DEFINE_boolean('start_training.save_train', default=True, help='')
    flags.DEFINE_boolean('start_training.save_val', default=True, help='')


def run(argv):
    del argv

    # get yaml file and print content
    cfg = YAMLConfig("../config_cotrain.yaml")
    print(cfg)

    # # retrieve value given a key
    # myoption = 'arch'
    # print("Set of parameter for {} key: ".format(myoption, cfg[myoption]))

    # # change specific parameter
    # parameter = 'name'
    # cfg[myoption][parameter] = 'segnet'
    # print(cfg)

    # # update specific key
    # param_dict = {'name': 'unet', 'num_classes': 4}
    # cfg.update({myoption: param_dict})

    # # update config file with new parameter
    # new_option = 'semi_training'
    # params_dict = {'loss': 'mse', 'lr': 0.002}
    # cfg.update(new_option=params_dict)

    # # delete parameter from paremeter
    # del cfg[new_option]
    # print(cfg)

    # get terminal arguments
    terminal_params = flags.FLAGS.flag_values_dict()
    initial = True
    before_opt = ''
    param_dict, tmp_dict = {}, {}
    for key, value in terminal_params.items():
        if key.__contains__('.'):
            # print(key)
            opt, param = key.split('.')
            # print(opt, param, value)
            tmp_dict = {param: value}

            if initial:
                before_opt = opt
                initial = False

            if opt == before_opt:
                param_dict.update(tmp_dict)
            else:
                # update yaml file
                cfg.update({before_opt: param_dict})
                param_dict = {param: value}
                before_opt = opt

    print()


if __name__ == '__main__':
    get_default_parameter()
    app.run(run)
