# TransRAC: Encoding Multi-scale Temporal Correlation with Transformers for Repetitive Action Counting
[![LICENSE](https://img.shields.io/badge/license-Anti%20996-blue.svg)](https://github.com/996icu/996.ICU/blob/master/LICENSE)
[![996.icu](https://img.shields.io/badge/link-996.icu-red.svg)](https://996.icu)
```
@inproceedings{hu2022transrac,
  author    = {Huazhang, Hu and
	       Sixun, Dong and
               Yiqun, Zhao and
               Zhengxin, Li and
               Dongze, Lian and
               Sheng hua, Gao},
  title     = {TransRAC: Encoding Multi-scale Temporal Correlation with Transformers for Repetitive Action Counting},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year      = {2022}
}
```
Official codes for CVPR 2022 paper "TransRAC: Encoding Multi-scale Temporal Correlation with Transformers for Repetitive Action Counting"
## Abstract 
Counting repetitive actions are widely seen in human activities such as physical exercise. Existing methods focus on performing repetitive action counting in short videos, which is tough for dealing with longer videos in more realistic scenarios. In the data-driven era, the degradation of such generalization capability is mainly attributed to the lack of long video datasets. To complement this margin, we introduce a new large-scale repetitive action counting dataset covering a wide variety of video lengths, along with more realistic situations where action interruption or action inconsistencies occur in the video. Besides, we also provide a fine-grained annotation of the action cycles instead of just counting annotation along with a numerical value. Such a dataset contains 1451 videos with about 20000 annotations, which is more challenging. For repetitive action counting towards more realistic scenarios, we further propose encoding multi-scale temporal correlation with transformers that can take into account both performance and efficiency. Furthermore, with the help of fine-grained annotation of action cycles, we propose a density map regression-based method to predict the action period, which yields better performance with sufficient interpretability. Our proposed method outperforms state-of-the-art methods on all datasets and also achieves better performance on the un-seen dataset without fine-tuning. Code and dataset will be released. 

## Neural Network architecture 
![architecture](https://github.com/SvipRepetitionCounting/SVIP_Counting/blob/hhz/figures/TransRAC_architecture.png)
[Network architecture](https://github.com/SvipRepetitionCounting/SVIP_Counting/blob/hhz/TransRAC.py)  

## News
- 2022-03-22: The [Repition Action Counting Dataset](https://svip-lab.github.io/dataset/RepCount_dataset.html) has released. 
- 2022-03-02: This paper has accepted on `CVPR 2022`


## Requirment 

### Enviroment
pytorch1.7.0  
mmcv-full 1.3.16  
cuda 11.4  
apex(recommand)  
cv2

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
pip install tqdm tensorboardX timm einops

```
## Train   
` python our_train.py `  
[train script](https://github.com/SvipRepetitionCounting/SVIP_Counting/blob/hhz/our_train.py)    

## [RepCount Dataset](https://svip-lab.github.io/dataset/RepCount_dataset.html)  
The homepage of [RepCount Dataset](https://svip-lab.github.io/dataset/RepCount_dataset.html) has released! Welcome to citing!   
### Dataset introduction  
We introduce a novel repetition action counting dataset called RepCount that contains videos with significant variations in length and allows for multiple kinds of anomaly cases. These video data collaborate with fine-grained annotations that indicate the beginning and end of each action period. Furthermore, the dataset consists of two subsets namely Part-A and Part-B. The videos in Part-A are fetched from YouTube, while the others in Part-B record simulated physical examinations by junior school students and teachers.   


We are constantly updating! 
Welcome to cite and follow our work!
If you have any questions, don't hesitate to contact us!
 






