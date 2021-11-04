# SVIP_Counting
Raw video-swin-transformer please click [raw video-swin-transformer](https://github.com/SwinTransformer/Video-Swin-Transformer)
Environment installing please refers to it.  

key enviroment pytorch1.7+  
cuda 10.2-11.0

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

`pip install kornia==0.5.0 ` 

  

### dataset 
LSP dataset  
total 962 videos 

### train  

` python rep_train.py ` 

[train script](https://github.com/SvipRepetitionCounting/SVIP_Counting/blob/hhz/rep_train.py)  

Neural Network structure  [Network structure](https://github.com/SvipRepetitionCounting/SVIP_Counting/blob/hhz/RepSwin.py)  



