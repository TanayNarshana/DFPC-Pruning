# [ICLR 2023] DFPC: Data flow driven pruning of coupled channels without data.
This repository is for the new deep neural network pruning method introduced in the following ICLR 2023 paper:
> **DFPC: Data flow driven pruning of coupled channels without data. [[Camera Ready](https://openreview.net/forum?id=mhnHqRqcjYU)]** \
> [Tanay Narshana](https://tanaynarshana.github.io/), [Chaitanya Murti](https://mllab.csa.iisc.ac.in/members/), and [Chiranjib Bhattacharyya](https://www.csa.iisc.ac.in/~chiru/) \
> Indian Institute of Science, Bengaluru, India.

TLDR: This paper introduces a novel method for pruning of networks containing coupled connections without using data, DFPC.

We posit "coupled connections" as a bottleneck to obtain lower inference latencies when pruning networks. We then provide a formalization to abstract "coupled connections" and use it to derive a data-free way to measure the importance of coupled neurons in a network. Our experimental results display the merit in pruning "coupled connections" for they obtain pruned models with a better latency-vs-accuracy.

We're working up to clean up our code and provide our models in a clean way. Everything should be up by mid June
Coming up..
1. ~~Data-free code for pruning mobilenets~~
2. Data-driven code for pruning resnets
3. ~~pruned models for resnet-50 in the data-driven regime.~~

Available Code:
1. Code for data-free experiments is available in the `Data-Free` folder.
2. Pruned models for the data-driven experiment for ResNet-50 on the ImageNet dataset is available in the `Pruned-Models` folder.

Feel free to contact us at `tanay.narshana@gmail.com`. (*Email is more recommended if you'd like quicker reply*)

## Reference
Please cite this in your publication if our work helps your research:

    @inproceedings{narshana2023dfpc,
    title={{DFPC}: Data flow driven pruning of coupled channels without data.},
    author={Tanay Narshana and Chaitanya Murti and Chiranjib Bhattacharyya},
    booktitle={The Eleventh International Conference on Learning Representations },
    year={2023},
    url={https://openreview.net/forum?id=mhnHqRqcjYU}
    }

## Acknowledgments
We are grateful for the code made available by [pytorch imagenet example](https://github.com/pytorch/examples/tree/master/imagenet), [Regularization-Pruning](https://github.com/MingSun-Tse/Regularization-Pruning), and [pytorch-summary](https://github.com/sksq96/pytorch-summary).


[![Visitors](https://api.visitorbadge.io/api/visitors?path=https%3A%2F%2Fgithub.com%2FTanayNarshana%2FDFPC-Pruning&countColor=%23263759)](https://visitorbadge.io/status?path=https%3A%2F%2Fgithub.com%2FTanayNarshana%2FDFPC-Pruning)
