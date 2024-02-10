# [ACR-Loss](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=96lS6HIAAAAJ&citation_for_view=96lS6HIAAAAJ:eQOLeE2rZwMC)

### https://aliprf.github.io/ACR-Loss/

### Accepted in ICPR 2022
ACR Loss: Adaptive Coordinate-based Regression Loss for Face Alignment

#### Link to the paper:
https://arxiv.org/pdf/2203.15835.pdf

https://ieeexplore.ieee.org/document/9956683

```diff
@@plaese STAR the repo if you like it.@@
```

```
Please cite this work as:

@INPROCEEDINGS{9956683,
  author={Fard, Ali Pourramezan and Mahoor, Mohammah H.},
  booktitle={2022 26th International Conference on Pattern Recognition (ICPR)}, 
  title={ACR Loss: Adaptive Coordinate-based Regression Loss for Face Alignment}, 
  year={2022},
  volume={},
  number={},
  pages={1807-1814},
  doi={10.1109/ICPR56361.2022.9956683}}
```

![Samples](https://github.com/aliprf/ACR-Loss/blob/main/img/ACR_300w_samples.png?raw=true)

## Introduction

Although deep neural networks have achieved reasonable accuracy in solving face alignment, it is still a challenging task, specifically when we deal with facial images, under occlusion, or extreme head poses. Heatmap-based Regression (HBR) and Coordinate-based Regression (CBR) are among the two mainly used methods for face alignment. CBR methods require less computer memory, though their performance is less than HBR methods. In this paper, we propose an Adaptive Coordinatebased Regression (ACR) loss to improve the accuracy of CBR for face alignment. Inspired by the Active Shape Model (ASM), we generate Smooth-Face objects, a set of facial landmark points with less variations compared to the ground truth landmark points. We then introduce a method to estimate the level of difficulty in predicting each landmark point for the network by comparing the distribution of the ground truth landmark points
and the corresponding Smooth-Face objects. Our proposed ACR Loss can adaptively modify its curvature and the influence of the loss based on the difficulty level of predicting each landmark point in a face. Accordingly, the ACR Loss guides the network toward challenging points than easier points, which improves the accuracy of the face alignment task. Our extensive evaluation shows the capabilities of the proposed ACR Loss in predicting facial landmark points in various facial images.

We evaluated our ACR Loss using MobileNetV2, EfficientNetB0, and EfficientNet-B3 on widely used 300W, and COFW datasets and showed that the performance of face alignment using the ACR Loss is much better than the widely-used L2 loss. Moreover, on the COFW dataset, we achieved state-of-theart accuracy. In addition, on 300W the ACR Loss performance is comparable to the state-of-the-art methods. We also compared the performance of MobileNetV2 trained using the ACR Loss with the lightweight state-of-the-art methods, and we achieved the best accuracy, highlighting the effectiveness of our ACR Loss for face alignment specifically for the lightweight models.


----------------------------------------------------------------------------------------------------------------------------------
## Installing the requirements
In order to run the code you need to install python >= 3.5. 
The requirements and the libraries needed to run the code can be installed using the following command:

```
  pip install -r requirements.txt
```


## Using the pre-trained models
You can test and use the preetrained models using the following codes:  
```
 tester = Test()
    tester.test_model(ds_name=DatasetName.w300,
                      pretrained_model_path='./pre_trained_models/ACRLoss/300w/EF_3/300w_EF3_ACRLoss.h5')

```


## Training Network from scratch


### Preparing Data
Data needs to be normalized and saved in npy format. 

### PCA creation
you can you the pca_utility.py class to create the eigenvalues, eigenvectors, and the meanvector:
```
pca_calc = PCAUtility()
    pca_calc.create_pca_from_npy(dataset_name=DatasetName.w300,
                                 labels_npy_path='./data/w300/normalized_labels/',
                                 pca_percentages=90)

```
### Training 
The training implementation is located in train.py class. You can use the following code to start the training:

```
    trainer = Train(arch=ModelArch.MNV2,
                    dataset_name=DatasetName.w300,
                    save_path='./')
```
