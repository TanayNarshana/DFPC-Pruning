# [ICLR 2023] DFPC: Data flow driven pruning of coupled channels without data.
[ICLR 2023] PyTorch Code to prune ResNets using DFPC: Data flow driven pruning of coupled channels without data.

## Step 1: Set up environment
- OS: Linux (Tested on Ubuntu 20.04. It should be all right for most linux platforms. Did not test on Windows and MacOS.)
- python=3.9.7 (conda is *strongly* suggested to manage environment)
- All the dependant libraries are summarized in `requirements.txt`.
- We use CUDA 10.2
```
conda create -n dfpc python=3.9.7
conda activate dfpc
conda install pip
pip install -r requirements.txt
```

## Step 2: Set up dataset
- We evaluate our methods on CIFAR-10/100. CIFAR-10/100 will be automatically downloaded during execution.

## Step 3: Download pretrained models
- Download pretrained models from [google drive](https://drive.google.com/drive/folders/1wGg3RJ2i-vAly4WWqZY3NbqgMGz9240V?usp=sharing) and put them in the `pretrained_checkpoints` folder.

## Step 4: Pruning ResNets
To prune ResNets, use the following command.
```
CUDA_VISIBLE_DEVICES=0 python main.py -a <model_name> --dataset <dataset_name> --scoring-strategy <strategy_name> --prune-coupled <pruning_coupled_channels>
```
Here, we can have the following options for arguments under the angled brackets.
- `<model_name>` can be replaced with `resnet50` or `resnet101`.
- `<dataset_name>` can be replaced with `cifar10` or `cifar100`.
- `<strategy_name>` can be replaced with either of `dfpc`, `l1`, or `random`.
- `<pruning_coupled_channels>` can be replaced with `0` or `1`. Setting this argument to 1 prunes both coupled and non-coupled channels in ResNets. Setting this argument to 0 prunes only the non-coupled channels.


## Acknowledgments
We are grateful for the code made available by [pytorch imagenet example](https://github.com/pytorch/examples/tree/master/imagenet), [Regularization-Pruning](https://github.com/MingSun-Tse/Regularization-Pruning), [pytorch-summary](https://github.com/sksq96/pytorch-summary), and [pytorch-cifar](https://github.com/kuangliu/pytorch-cifar)