# TransRAC: Encoding Multi-scale Temporal Correlation with Transformers for Repetitive Action Counting（CVPR 2022 Oral）
[![LICENSE](https://img.shields.io/badge/license-Anti%20996-blue.svg)](https://github.com/996icu/996.ICU/blob/master/LICENSE)
[![996.icu](https://img.shields.io/badge/link-996.icu-red.svg)](https://996.icu)

Here is the official implementation for CVPR 2022 paper "TransRAC: Encoding Multi-scale Temporal Correlation with Transformers for Repetitive Action Counting" 


## 🌱News
- 2025-02-06: After contacting RepNet's authors, we clarified the different performances caused by experimental settings and the evaluation details between RepNet and TransRAC. Furthermore, we wrote a short note:[ Detailed Explanation of the Experimental Settings in TransRAC.pdf ](https://github.com/SvipRepetitionCounting/TransRAC/blob/main/TranRAC_Experimental_Settings.pdf) to help address the confusion.
- 2024-11-14: We noticed the authors of RepNet posted a note titled "A Short Note on Evaluating RepNet for Temporal Repetition Counting in Videos."  We are writing a short paper with a detailed explanation of our experimental setting. In a word, we evaluate different frameworks by retraining them on RepCount-A.
- 2023-07-13: ~~We are planning to release the RepCount-B dataset within a week.~~ Sorry, we can not release original videos of Part B. Please refer to [Issue 44](https://github.com/SvipRepetitionCounting/TransRAC/issues/44) for detailed reasons and solutions.
- 2023-04-10: We have updated the Chinese introduction of the paper. [[Zhihu](https://zhuanlan.zhihu.com/p/543376943?)]
- 2022-07-18: The model ckpt has been available.[[OneDrive(extraction code: transrac)](https://shanghaitecheducn-my.sharepoint.com/:f:/g/personal/dongsx_shanghaitech_edu_cn/Eg2-I2dG_BhKkuBJGnTg200BhhsEAYmCx3xgAvRuTEURuA?e=YURfkP)][[BaiduDisk(extraction code: 2022)](https://pan.baidu.com/s/13pVq7JVsaM9MrJ-AsO5Lvw?pwd=2022)]
- 2022-06-24: We are invited to oral presentation with virtual attendance. 
- 2022-06-01: The oral presentation of our work is available. [[Youtube](https://youtu.be/SFpUS9mHHpk)] [[Bilibili](https://www.bilibili.com/video/BV1B94y1S7oP?share_source=copy_web)]
- 2022-04-05: The preprint of the paper is available. [[Paper](https://arxiv.org/abs/2204.01018)]
- 2022-03-22: The Repetition Action Counting **Dataset Homepage** is open for the community. [[Homepage](https://svip-lab.github.io/dataset/RepCount_dataset.html)]
- 2022-03-02: This paper has been accepted by **`CVPR 2022`** as  **`Oral presentation`**

## Introduction
Counting repetitive actions are widely seen in human activities such as physical exercise. Existing methods focus on performing repetitive action counting in short videos, which is tough for dealing with longer videos in more realistic scenarios. In the data-driven era, the degradation of such generalization capability is mainly attributed to the lack of long video datasets. To complement this margin, we introduce a new large-scale repetitive action counting dataset covering a wide variety of video lengths, along with more realistic situations where action interruption or action inconsistencies occur in the video. Besides, we also provide a fine-grained annotation of the action cycles instead of just counting annotation along with a numerical value. Such a dataset contains 1451 videos with about 20000 annotations, which is more challenging. For repetitive action counting towards more realistic scenarios, we further propose **encoding multi-scale temporal correlation with transformers** that can take into account both performance and efficiency. Furthermore, with the help of fine-grained annotation of action cycles, we propose a density map regression-based method to predict the action period, which yields better performance with sufficient interpretability. Our proposed method outperforms state-of-the-art methods on all datasets and also achieves better performance on the un-seen dataset without fine-tuning. 




## RepCount Dataset   
The Homepage of [RepCount Dataset](https://svip-lab.github.io/dataset/RepCount_dataset.html) is available now. 

<table rules="none" align="center">
	<tr>
		<td>
			<center>
				<img src="https://github.com/SvipRepetitionCounting/TransRAC/blob/main/figures/raising.gif" width="100%" />
      </center>
		</td>
		<td>
			<center>
				<img src="https://github.com/SvipRepetitionCounting/TransRAC/blob/main/figures/jump_jack.gif" width="100%" />
      </center>
		</td>
  </tr>
  <tr>
		<td>
			<center>
				<img src="https://github.com/SvipRepetitionCounting/TransRAC/blob/main/figures/squat.gif" width="100%" />
      </center>
		</td>
    <td>
			<center>
				<img src="https://github.com/SvipRepetitionCounting/TransRAC/blob/main/figures/pull_up.gif" width="100%" />
			</center>
		</td>
	</tr>
</table>


### Dataset introduction  
We introduce a novel repetition action counting dataset called RepCount that contains videos with significant variations in length and allows for multiple kinds of anomaly cases. These video data collaborate with fine-grained annotations that indicate the beginning and end of each action period. Furthermore, the dataset consists of two subsets namely Part-A and Part-B. The videos in Part-A are fetched from YouTube, while the others in Part-B record simulated physical examinations by junior school students and teachers.   

## Video Presentation  
<center><a href="https://www.bilibili.com/video/BV1B94y1S7oP?share_source=copy_web" target="_blank" style="color: #990000"> Bilibili </a></center>       <br/> 
<center><a href="https://youtu.be/SFpUS9mHHpk" target="_blank" style="color: #990000"> YouTube </a></center>  

------
## Usage  
### Install 
Please refer to [install.md](https://github.com/SvipRepetitionCounting/TransRAC/blob/main/install.md) for installation.

### Data preparation
Firstly, you should loading the pretrained model [Video Swin Transformer](https://github.com/SwinTransformer/Video-Swin-Transformer)([github](https://github.com/SwinTransformer/storage/releases/download/v1.0.4/swin_tiny_patch244_window877_kinetics400_1k.pth)) in to the folder '[pretrained](https://github.com/SvipRepetitionCounting/TransRAC/tree/main/pretrained)'.

Secondly, you should modify [train.py](https://github.com/SvipRepetitionCounting/TransRAC/blob/main/train.py) to your config.

*Tips*: The data form can be .mp4 or .npz. We recommend to use .npz data because it is faster. We will upload the preprocessed data(.npz) soon. You can also refer to [video2npz](https://github.com/SvipRepetitionCounting/TransRAC/blob/main/tools/video2npz.py) to transform them by yourself.


### Train   
` python train.py `    

### Model Zoo
##### RepCount Dataset 
|  Method   | Backbone | Frame | Training Dataset | CheckPoint |  MAE  | OBO |
|  :---: | :-----: | :----:  | :----:       |  :---------------------------:            | :---: | :---: |
| Ours  | [Video Swin Transformer](https://github.com/SwinTransformer/Video-Swin-Transformer) | 64 | [RepCount-A](https://svip-lab.github.io/dataset/RepCount_dataset.html) | [OneDrive(extraction code: transrac)](https://shanghaitecheducn-my.sharepoint.com/:f:/g/personal/dongsx_shanghaitech_edu_cn/Eg2-I2dG_BhKkuBJGnTg200BhhsEAYmCx3xgAvRuTEURuA?e=YURfkP) / [BaiduDisk(extraction code: 2022)](https://pan.baidu.com/s/13pVq7JVsaM9MrJ-AsO5Lvw?pwd=2022) | 0.44 | 0.29 |

We will upload more TransRAC trained model soon which may help you.

------


If you have any questions, don't hesitate to contact us!

But please understand that the response may be delayed as we are working on other research.😖

------
## Citation 
If you find this project or dataset useful, please consider citing the paper.
```
@inproceedings{hu2022transrac,
  title={TransRAC: Encoding Multi-scale Temporal Correlation with Transformers for Repetitive Action Counting},
  author={Hu, Huazhang and Dong, Sixun and Zhao, Yiqun and Lian, Dongze and Li, Zhengxin and Gao, Shenghua},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={19013--19022},
  year={2022}
}
```


