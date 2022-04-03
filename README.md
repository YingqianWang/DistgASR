## DistgASR: Disentangling Mechanism for Light Field Angular Super-Resolution
<br>

**This is the PyTorch implementation of the angular SR method in our paper "*Disentangling Light Fields for Super-Resolution and Disparity Estimation*". Please refer to our [paper](https://arxiv.org/pdf/2202.10603.pdf) and [project page](https://yingqianwang.github.io/DistgLF) for details.**<br><br>

## Network Architecture:
<p align="center"> <img src="https://raw.github.com/YingqianWang/DistgASR/master/Figs/DistgASR.png" width="90%"> </p>
<br><br>

## Codes and Models:

### Requirement:
* **PyTorch 1.3.0, torchvision 0.4.1. The code is tested with python=3.6, cuda=9.0.**
* **Matlab (For training/test data generation and performance evaluation)**

### Datasets:
The datasets used in our paper can be downloaded through [this link](https://stuxidianeducn-my.sharepoint.com/:f:/g/personal/zyliang_stu_xidian_edu_cn/EklAQ0a4ftJLvEfjZ64UoWgBd5he4N37_VSM9u41XfocDQ).

### Train:
* Run **`Generate_Data_for_Training.m`** to generate training data.
* Run `train.py` to perform network training.
* Checkpoint will be saved to **`./log/`.

### Test:
* Run `Generate_Data_for_Test.m` to generate test data.
* Run `test.py` to perform network inference.
* The PSNR and SSIM values of each dataset will be saved to `./log/`.
<br><br>

## Results:

### Quantitative Results:
<p align="center"> <img src="https://raw.github.com/YingqianWang/DistgASR/master/Figs/QuantitativeASR.png" width="65%"> </p>

### Visual Comparisons:
<p align="center"> <img src="https://raw.github.com/YingqianWang/DistgASR/master/Figs/Visual-ASR.png" width="100%"> </p>

### Angular Consistency:
<p align="center"> <a href="https://wyqdatabase.s3.us-west-1.amazonaws.com/DistgLF-AngularSR.mp4"><img src="https://raw.github.com/YingqianWang/DistgASR/master/Figs/AngCons-ASR.png" width="80%"></a> </p>


## Citiation
**If you find this work helpful, please consider citing:**
```
@Article{DistgLF,
    author    = {Wang, Yingqian and Wang, Longguang and Wu, Gaochang and Yang, Jungang and An, Wei and Yu, Jingyi and Guo, Yulan},
    title     = {Disentangling Light Fields for Super-Resolution and Disparity Estimation},
    journal   = {IEEE TPAMI}, 
    year      = {2022},   
}
```
<br>

## Contact
**Welcome to raise issues or email to [wangyingqian16@nudt.edu.cn](wangyingqian16@nudt.edu.cn) for any question regarding this work.**
