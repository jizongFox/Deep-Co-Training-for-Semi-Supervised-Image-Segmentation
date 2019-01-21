#!/usr/bin/env bash
max_peoch=100
data_aug=None
logdir=cardiac/unet_No_agument
mkdir -p archives/$logdir
## Fulldataset baseline

#currentfoldername=FS_fulldata
#rm -rf $logdir/$currentfoldername
#python train.py Trainer.save_dir=runs/$logdir/$currentfoldername Trainer.max_epoch=$max_peoch \
#Dataset.augment=$data_aug Dataloader.batch_sampler="['PatientSampler',{'grp_regex':'(patient\d+_\d+)_\d+','shuffle':False}]"
#rm -rf archives/$logdir/$currentfoldername
#mv -f runs/$logdir/$currentfoldername archives/$logdir

## Partial dataset baseline
#currentfoldername=FS_partial
#rm -rf $logdir/$currentfoldername
#python train_cotraining.py Trainer.save_dir=runs/$logdir/$currentfoldername Trainer.max_epoch=$max_peoch \
#Dataset.augment=$data_aug StartTraining.train_jsd=False StartTraining.train_adv=False Dataloader.batch_sampler=Dataloader.batch_sampler="['PatientSampler',{'grp_regex':'(patient\d+_\d+)_\d+','shuffle':False}]"
#
#rm -rf archives/$logdir/$currentfoldername
#mv -f runs/$logdir/$currentfoldername archives/$logdir


currentfoldername=only_jsd
rm -rf $logdir/$currentfoldername
python train_cotraining.py Trainer.save_dir=runs/$logdir/$currentfoldername Trainer.max_epoch=$max_peoch \
Dataset.augment=$data_aug StartTraining.train_jsd=True StartTraining.train_adv=False \
Dataloader.batch_sampler=Dataloader.batch_sampler="['PatientSampler',{'grp_regex':'(patient\d+_\d+)_\d+','shuffle':False}]"

rm -rf archives/$logdir/$currentfoldername
mv -f runs/$logdir/$currentfoldername archives/$logdir

currentfoldername=only_adv
rm -rf $logdir/$currentfoldername
python train_cotraining.py Trainer.save_dir=runs/$logdir/$currentfoldername Trainer.max_epoch=$max_peoch \
Dataset.augment=$data_aug StartTraining.train_jsd=False StartTraining.train_adv=True \
Dataloader.batch_sampler=Dataloader.batch_sampler="['PatientSampler',{'grp_regex':'(patient\d+_\d+)_\d+','shuffle':False}]"

rm -rf archives/$logdir/$currentfoldername
mv -f runs/$logdir/$currentfoldername archives/$logdir

currentfoldername=jsd_adv
rm -rf $logdir/$currentfoldername
python train_cotraining.py Trainer.save_dir=runs/$logdir/$currentfoldername Trainer.max_epoch=$max_peoch \
Dataset.augment=$data_aug StartTraining.train_jsd=True StartTraining.train_adv=True \
Dataloader.batch_sampler=Dataloader.batch_sampler="['PatientSampler',{'grp_regex':'(patient\d+_\d+)_\d+','shuffle':False}]"

rm -rf archives/$logdir/$currentfoldername
mv -f runs/$logdir/$currentfoldername archives/$logdir


python generalframework/postprocessing/plot.py --folders archives/$logdir/FS_fulldata/ archives/$logdir/FS_partial/ archives/$logdir/only_jsd/ archives/$logdir/only_adv/ archives/$logdir/jsd_adv/ --file val_dice.npy --axis 1 2 3 --postfix=test
python generalframework/postprocessing/plot.py --folders archives/$logdir/FS_fulldata/ archives/$logdir/FS_partial/ archives/$logdir/only_jsd/ archives/$logdir/only_adv/ archives/$logdir/jsd_adv/ --file val_batch_dice.npy --axis 1 2 3 --postfix=test