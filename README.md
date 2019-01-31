# [PYTORCH] YOLO (You Only Look Once)

## Introduction

Here is my pytorch implementation of the model described in the paper **YOLO9000: Better, Faster, Stronger** [paper](https://arxiv.org/abs/1612.08242). 
<p align="center">
  <img src="demo/video.gif"><br/>
  <i>An example of my model's output.</i>
</p>

## How to use my code

With my code, you can:
* **Train your model from scratch**
* **Train your model with my trained model**
* **Evaluate test images with either my trained model or yours**

## Requirements:

* **python 3.6**
* **pytorch 0.4**
* **opencv (cv2)**
* **tensorboard**
* **tensorboardX** (This library could be skipped if you do not use SummaryWriter)
* **numpy**

## Datasets:

I used 4 different datases: VOC2007, VOC2012, COCO2014 and COCO2017. Statistics of datasets I used for experiments is shown below

| Dataset                | Classes | #Train images/objects | #Validation images/objects |
|------------------------|:---------:|:-----------------------:|:----------------------------:|
| VOC2007                |    20   |      5011/12608       |           4952/-           |
| VOC2012                |    20   |      5717/13609       |           5823/13841       |
| COCO2014               |    80   |         83k/-         |            41k/-           |
| COCO2017               |    80   |         118k/-        |             5k/-           |

Create a data folder under the repository,

```
cd {repo_root}
mkdir data
```

- **VOC**:
  Download the voc images and annotations from [VOC2007](http://host.robots.ox.ac.uk/pascal/VOC/voc2007) or [VOC2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012). Make sure to put the files as the following structure:
  ```
  VOCDevkit
  ├── VOC2007
  │   ├── Annotations  
  │   ├── ImageSets
  │   ├── JPEGImages
  │   └── ...
  └── VOC2012
      ├── Annotations  
      ├── ImageSets
      ├── JPEGImages
      └── ...
  ```
  
- **COCO**:
  Download the coco images and annotations from [coco website](http://cocodataset.org/#download). Make sure to put the files as the following structure:
  ```
  COCO
  ├── annotations
  │   ├── instances_train2014.json
  │   ├── instances_train2017.json
  │   ├── instances_val2014.json
  │   └── instances_val2017.json
  │── images
  │   ├── train2014
  │   ├── train2017
  │   ├── val2014
  │   └── val2017
  └── anno_pickle
      ├── COCO_train2014.pkl
      ├── COCO_val2014.pkl
      ├── COCO_train2017.pkl
      └── COCO_val2017.pkl
  ```
  
## Setting:

* **Model structure**: In compared to the paper, I changed structure of top layers, to make it converge better. You could see the detail of my YoloNet in **src/yolo_net.py**.
* **Data augmentation**: I performed dataset augmentation, to make sure that you could re-trained my model with small dataset (~500 images). Techniques applied here includes HSV adjustment, crop, resize and flip with random probabilities
* **Loss**: The losses for object and non-objects are combined into a single loss in my implementation
* **Optimizer**: I used SGD optimizer and my learning rate schedule is as follows: 

|         Epoches        | Learning rate |
|------------------------|:---------------:|
|          0-4           |      1e-5     |
|          5-79          |      1e-4     |
|          80-109        |      1e-5     |
|          110-end       |      1e-6     |

* In my implementation, in every epoch, the model is saved only when its loss is the lowest one so far. You could also use early stopping, which could be triggered by specifying a positive integer value for parameter **es_patience**, to stop training process when validation loss has not been improved for **es_patience** epoches.

## Trained models

You could find all trained models I have trained in [YOLO trained models](https://drive.google.com/open?id=1Ee6FHQTGuJpNRYSa8DtHWzu4yWNyc7sp)

## Training

For each dataset, I provide 2 different pre-trained models, which I trained with corresresponding dataset:
- **whole_model_trained_yolo_xxx**: The whole trained model.
- **only_params_trained_yolo_xxx**: The trained parameters only.

You could specify which trained model file you want to use, by the parameter **pre_trained_model_type**. The parameter **pre_trained_model_path** then is the path to that file.

If you want to train a model with a VOC dataset, you could run:
- **python3 train_voc.py --year year**: For example, python3 train_voc.py --year 2012

If you want to train a model with a COCO dataset, you could run:
- **python3 train_coco.py --year year**: For example, python3 train_coco.py --year 2014

If you want to train a model with both COCO datasets (training set = train2014 + val2014 + train2017, val set = val2017), you could run:
- **python3 train_coco_all.py**

## Test

For each type of dataset (VOC or COCO), I provide 3 different test scripts:

If you want to test a trained model with a standard VOC dataset, you could run:
- **python3 test_xxx_dataset.py --year year**: For example, python3 test_coco_dataset.py --year 2014

If you want to test a model with some images, you could put them into the same folder, whose path is **path/to/input/folder**, then run:
- **python3 test_xxx_images.py --input path/to/input/folder --output path/to/output/folder**: For example, python3 train_voc_images.py --input test_images --output test_images

If you want to test a model with a video, you could run :
- **python3 test_xxx_video.py --input path/to/input/file --output path/to/output/file**: For example, python3 test_coco_video --input test_videos/input.mp4 --output test_videos/output.mp4

## Experiments:

I trained models in 2 machines, one with NVIDIA TITAN X 12gb GPU and the other with NVIDIA quadro 6000 24gb GPU.

The training/test loss curves for each experiment are shown below:

- **VOC2007**
![voc2007 loss](demo/voc2007.png) 
- **VOC2012**
![voc2012 loss](demo/voc2012.png)
- **COCO2014**
![coco2014 loss](demo/coco2014.png)
- **COCO2014+2017**
![coco2014_2017 loss](demo/coco2014_2017.png)

Statistics for mAP will be updated soon ...

## Results

Some output predictions for experiments for each dataset are shown below:

- **VOC2007**

<img src="demo/voc2007_1.jpg" width="280"> <img src="demo/voc2007_2.jpg" width="280"> <img src="demo/voc2007_3.jpg" width="280">

- **VOC2012**

<img src="demo/voc2012_1.jpg" width="280"> <img src="demo/voc2012_2.jpg" width="280"> <img src="demo/voc2012_3.jpg" width="280">

- **COCO2014**

<img src="demo/coco2014_1.jpg" width="280"> <img src="demo/coco2014_2.jpg" width="280"> <img src="demo/coco2014_3.jpg" width="280">

- **COCO2014+2017**

<img src="demo/coco2014_2017_1.jpg" width="280"> <img src="demo/coco2014_2017_2.jpg" width="280"> <img src="demo/coco2014_2017_3.jpg" width="280">
