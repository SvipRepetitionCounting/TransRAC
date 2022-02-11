# TransRAC
##  Encoding Multi-scale Temporal Correlation with Transformers for Repetitive Action Counting

### train  
` python our_train.py ` 

[train script](https://github.com/SvipRepetitionCounting/SVIP_Counting/blob/hhz/our_train.py)  

### Neural Network structure   
[Network structure](https://github.com/SvipRepetitionCounting/SVIP_Counting/blob/hhz/TransRAC.py)  

### enviroment   

pytorch1.7.0
mmcv 1.3.16
cuda 11.4

### install mmcv
```shell
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7.0/index.html
```

See [here](https://github.com/open-mmlab/mmcv#installation) for different versions of MMCV compatible to different PyTorch and CUDA versions.

### Clone the MMAction2 repository.

```shell
git clone https://github.com/open-mmlab/mmaction2.git
cd mmaction2
```

Install build requirements and then install MMAction2.

```shell
pip install -r requirements/build.txt
pip install -v -e .  # or "python setup.py develop"
```

### other dependency
```
pip install kornia==0.5.0  

pip install tqdm tensorboardX timm einops

```

### RepCount dataset 
you can download data from https://anonymous.4open.science/r/RepCount/README.md  
RepCount dataset include:
>total 962 videos  
>test 124 videos   

We are constantly updating! 






