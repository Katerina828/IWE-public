# IWE
IWE: In-distribution watermark embedding for DNNs

先安装toolbox，链接： https://github.com/xrj-com/marveltoolbox

To train the clean/watermarked model, run `train_cifar_mean.py` by the following line.

```
python train_cifar_mean.py 
```
There are some arguments in train_cifar_mean.py, for examples,

- To train clean models, set delta = 0.0;

- To train watermarked model, set delta = 0.01.

Dataset默认存储文件夹是

CIFAR10:'/workspace/DATASET/CIFAR10'，

如果需要修改，请去'hat_datasets/configs.py'

