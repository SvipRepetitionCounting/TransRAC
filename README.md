# TransRAC: Encoding Multi-scale Temporal Correlation with Transformers for Repetitive Action Counting（CVPR 2022 Oral）
[![LICENSE](https://img.shields.io/badge/license-Anti%20996-blue.svg)](https://github.com/996icu/996.ICU/blob/master/LICENSE)
[![996.icu](https://img.shields.io/badge/link-996.icu-red.svg)](https://996.icu)

Here is the official implementation for CVPR 2022 paper "TransRAC: Encoding Multi-scale Temporal Correlation with Transformers for Repetitive Action Counting"


## 🌱News
- 2022-04-05: The [arXiv preprint](https://arxiv.org/abs/2204.01018) of the paper is available now. 
- 2022-03-22: The [Repition Action Counting Dataset](https://svip-lab.github.io/dataset/RepCount_dataset.html) Homepage is open for the community. 
- 2022-03-02: This paper has been accepted by **`CVPR 2022`** as  **`Oral presentation`**

## Introduction
Counting repetitive actions are widely seen in human activities such as physical exercise. Existing methods focus on performing repetitive action counting in short videos, which is tough for dealing with longer videos in more realistic scenarios. In the data-driven era, the degradation of such generalization capability is mainly attributed to the lack of long video datasets. To complement this margin, we introduce a new large-scale repetitive action counting dataset covering a wide variety of video lengths, along with more realistic situations where action interruption or action inconsistencies occur in the video. Besides, we also provide a fine-grained annotation of the action cycles instead of just counting annotation along with a numerical value. Such a dataset contains 1451 videos with about 20000 annotations, which is more challenging. For repetitive action counting towards more realistic scenarios, we further propose **encoding multi-scale temporal correlation with transformers** that can take into account both performance and efficiency. Furthermore, with the help of fine-grained annotation of action cycles, we propose a density map regression-based method to predict the action period, which yields better performance with sufficient interpretability. Our proposed method outperforms state-of-the-art methods on all datasets and also achieves better performance on the un-seen dataset without fine-tuning. 

![architecture](https://github.com/SvipRepetitionCounting/TransRAC/blob/main/figures/TransRAC_architecture.png)


## RepCount Dataset   
The Homepage of [RepCount Dataset](https://svip-lab.github.io/dataset/RepCount_dataset.html) is available now. 
![RepCount](https://github.com/svip-lab/svip-lab.github.io/blob/master/img/dataset/RepCount_dataset/1.jpg)

### Dataset introduction  
We introduce a novel repetition action counting dataset called RepCount that contains videos with significant variations in length and allows for multiple kinds of anomaly cases. These video data collaborate with fine-grained annotations that indicate the beginning and end of each action period. Furthermore, the dataset consists of two subsets namely Part-A and Part-B. The videos in Part-A are fetched from YouTube, while the others in Part-B record simulated physical examinations by junior school students and teachers.   

------
## Usage  
### Install 
Please refer to [install.md](https://github.com/SvipRepetitionCounting/TransRAC/blob/main/install.md) for installation.

### Train   
` python train.py `    

------


If you have any questions, don't hesitate to contact us!

If you find the project or the dataset is useful, please consider citing the paper.

```
@article{hu2022transrac,
  title={TransRAC: Encoding Multi-scale Temporal Correlation with Transformers for Repetitive Action Counting},
  author={Hu, Huazhang and Dong, Sixun and Zhao, Yiqun and Lian, Dongze and Li, Zhengxin and Gao, Shenghua},
  journal={arXiv preprint arXiv:2204.01018},
  year={2022}
}
```



