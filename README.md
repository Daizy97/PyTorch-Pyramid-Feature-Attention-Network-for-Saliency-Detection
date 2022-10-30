# Generate Saliency maps for [Pyramid Feature Attention Network for Saliency Detection](http://openaccess.thecvf.com/content_CVPR_2019/html/Zhao_Pyramid_Feature_Attention_Network_for_Saliency_Detection_CVPR_2019_paper.html), CVPR 2019

## Install Dependencies
The code is written in Python 3.6 using the following libraries:
```
numpy
tqdm
opencv-python
torch==1.1.0
torchvision==0.3.0
```
Install the libraries using [requirements.txt](requirements.txt) as:
```
pip install -r requirements.txt
```

## Data
We use ImageNet dataset.

## Pre-trained Model
Download the pre-trained model from [Google Drive](https://drive.google.com/file/d/1Sc7dgXCZjF4wVwBihmIry-Xk7wTqrJdr/view?usp=sharing).

## Usage
Use the command below to generate and save saliency maps with different thresholds:
```
python inference.py --save_img --sal_threshold 0.1
```



## Reference
Keras implementation : [LINK](https://github.com/CaitinZhao/cvpr2019_Pyramid-Feature-Attention-Network-for-Saliency-detection)
