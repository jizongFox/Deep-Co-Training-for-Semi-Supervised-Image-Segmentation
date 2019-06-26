This is the code using ADMM for weaky supervised setting with ACDC and Promise dataset
----

Author: Jizong Peng
Data: 18.11.2018

The code here is for the paper "DCSegNet: A Discretely-Constrained Segmentation Network for Medical Segmentation", accepted by Medical Imaging meets NIPS, NeurIPS 2018, Montreal.

Before running script, install the package by ``pip install -e .``


For supervised training using ACDC dataset:<br>
``python train.py --dataroot=cardiac --method=fullysupervised --data_aug=True --arch=enet --loss=cross_entropy --max_epoch=1 --save_dir=results/cardiac/FS_enet_Daug --num_admm_innerloop=1
``
For weakly-supervised training with ACDC dataset and ADMM_size method:<br>
``python train.py --dataroot=cardiac --method=admm_size --data_aug=True --arch=enet  --max_epoch=1 --save_dir=results/cardiac/size_enet_Daug_0.0 --eps=0.0
``
Where the ``eps`` is the size error ratio between `0` and `1`.

You can set a global size constraint by using the tag: ``--individual_size_constraint=False`` and `--global_upbound=20`, `global_lowbound==2000`, as well.

For a training using GC and size together: <br>
`python train.py --dataroot=cardiac --method=admm_gc_size --data_aug=True --arch=enet --max_epoch=1 --save_dir=results/cardiac/gcsize_enet_Daug_0.0 --eps=0.0`


For ADMM training with inequality constraints, using 
`python train_in.py --dataroot=cardiac --method=admm_gc_size_in --data_aug=True --arch=enet --max_epoch=1 --save_dir=results/cardiac/gcsizeIN_enet_Daug_0.0 --eps=0.0`

##Attention

Each script can cost 3 days for a whole training and the results can be different with different initializations. This ADMM way is time costing and relatively hard to tune. The hyparameters are set as a grid search of using ``parameterSearch.py`` given `--name` and `--output_dir`.


Good luck with the fine-tuning and Happy new Year.

Jizong
