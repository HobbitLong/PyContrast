### Data prepartion
Download the ImageNet dataset from http://www.image-net.org/, 
and move validation images into subfolders using this 
[shell script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh).

### Training
Now it only supports **DistributedDataParallel** training (single-node multi-GPU or 
multi-node multi-GPU). It has predefined the configurations for several methods. 
For example, the training command of `CMC` is (take single machine with 8 GPUs as an example):
```
python main_contrast.py \
  --method CMC \
  --data_folder /path/to/data \
  --multiprocessing-distributed --world-size 1 --rank 0 \
```
You can replace `CMC` with `InsDis`, `MoCo`, `MoCov2`, `PIRL`, `InfoMin` for predefined 
methods. You may also customize your own model using `--method Customize` with the following 
options, see [options/base_options](../options/base_options.py) for details. You can use 
`--arch` option to specify different models

(optional) If you want to use mixed precision training, please appending the following options:  
```
--amp --opt_level O1
```

(Optional) If you would like to use multi-node for training, the example command is:
```
# node 1
python main_contrast.py --method CMC --batch_size 512 -j 40 --learning_rate 0.06 --multiprocessing-distributed --dist-url 'tcp://10.128.0.4:12345' --world-size 2 --rank 0
# node 2
python main_contrast.py --method CMC --batch_size 512 -j 40 --learning_rate 0.06 --multiprocessing-distributed --dist-url 'tcp://10.128.0.4:12345' --world-size 2 --rank 1
```
where the `--batch_size` means global batch size and `-j` indicates number of workers on each node. Node that currently intermediate checkpoints will
be saved on both machines. This is an easy setting for my case. If you have NSF shared by nodes, then you may need to do a simple modification
[here](https://github.com/HobbitLong/PyContrast/blob/master/pycontrast/learning/contrast_trainer.py#L110), replace `local_rank` with `rank`, 
which hopefully should be configurable through commands in future updates.

### Linear classifier evaluation
Exemplar command is:
```
python main_linear.py --method CMC \ 
  --ckpt model.pth \
  --aug_linear RA \
  --multiprocessing-distributed --world-size 1 --rank 0 \
```
which uses `RandAugment` for data augmentation, if you want to use the plain data augmentation, replace
`--aug_linear RA` with `--aug_linear NULL`. 