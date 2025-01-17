# POLO-NECIL
Non-Exemplar Class-Incremental Learning via Adaptive Old Class Reconstruction[ACM MM 23]

Paper link: https://dl.acm.org/doi/10.1145/3581783.3611926 


## Abstract
> In the Class-Incremental Learning (CIL) task, rehearsal-based approaches have received a lot of attention recently. However, storing old class samples is often infeasible in application scenarios where device memory is insufficient or data privacy is important. Therefore, it is necessary to rethink Non-Exemplar Class-Incremental Learning (NECIL). In this paper, we propose a novel NECIL method named POLO with an adaPtive Old cLass recOnstruction mechanism, in which a density-based prototype reinforcement method (DBR), a topology-correction prototype adaptation method (TPA), and an adaptive prototype augmentation method (APA) are designed to reconstruct pseudo features of old classes in new incremental sessions. Specifically, the DBR focuses on the low-density features to maintain the model’s discriminative ability for old classes. Afterward, the TPA is designed to adapt old class prototypes to new feature spaces in the incremental learning process. Finally, the APA is developed to further adapt pseudo feature spaces of old classes to new feature spaces. Experimental evaluations on four benchmark datasets demonstrate the effectiveness of our proposed method over the state-of-the-art NECIL methods.

## Framework

![image](https://github.com/Mysteriousplayer/POLO-NECIL/blob/main/model_v8.png)

-We propose a novel adaPtive Old cLass recOnstruction method (POLO) for the NECIL problem, which is a three-step paradigm containing prototype reinforcement, prototype adaption, and adaptive prototype augmentation.

-We design a density-based prototype reinforcement (DBR) method for prototype reinforcement, which focuses on the low-density features to maintain the model’s discriminative ability for old classes.

-We design a topology-correction prototype adaptation (TPA) method to adapt old class prototypes to new feature spaces and develop an adaptive prototype augmentation (APA) method to further adapt pseudo feature spaces of old classes to new feature spaces in the incremental learning process. 

## Datasets
CIFAR-100, Imagenet-100, Tiny-Imagenet, and  Imagenet-1000. 

## Installation
Install all requirements required to run the code on a Python 3.x by:
> First, activate a new conda environment.
> 
> pip install -r requirements.txt

## Training
All commands should be run under the project root directory. 

```
polo_cifar:
sh run_b50c10.sh
sh run_b50c5.sh
sh run_b40c3.sh

polo_sub:
sh run_b50c10_sub.sh
sh run_b50c5_sub.sh

 
polo_tiny:
sh run_b100c20.sh
sh run_b100c10.sh
sh run_b100c5.sh

```
## Results
Results will be saved in log/ and model_saved_check/. 

Model link: 

## Citation
If you found our work useful for your research, please cite our work:
```
@inproceedings{polo,
author = {Wang, Shaokun and Shi, Weiwei and He, Yuhang and Yu, Yifan and Gong, Yihong},
title = {Non-Exemplar Class-Incremental Learning via Adaptive Old Class Reconstruction},
year = {2023},
booktitle = {Proceedings of the 31st ACM International Conference on Multimedia},
pages = {4524–4534},
numpages = {11}
}
```

## Acknowledgments
We thank the following repo providing helpful functions in our work. 

PASS: https://github.com/Impression2805/CVPR21_PASS
