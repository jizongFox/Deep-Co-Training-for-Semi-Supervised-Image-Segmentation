#!/usr/bin/env bash
cd ../..
python train_mean_teacher.py \
Trainer.save_dir=runs/mean_teacher_GMBaseline \
Trainer.max_epoch=300 Trainer.axises=[0,1] \
Dataset.augment=PILaugment Dataset.transform="segment_transform((256,256))" \
Arch.name=enet Arch.num_classes=2  \
Dataset.root_dir=dataset/GM_Challenge \
Adv_Scheduler.max_value=0.05 \
StartTraining.augment_labeled_data=True \
StartTraining.augment_unlabeled_data=True \
Lab_Dataloader.batch_sampler=None