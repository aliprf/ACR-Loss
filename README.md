# ACR-Loss
ACR Loss: Adaptive Coordinate-based Regression Loss for Face Alignment

#### Link to the paper:
https://arxiv.org/pdf/2203.15835.pdf



## Introduction

Although deep neural networks have achieved reasonable accuracy in solving face alignment, it is still a challenging task, specifically when we deal with facial images, under occlusion, or extreme head poses. Heatmap-based Regression (HBR) and Coordinate-based Regression (CBR) are among the two mainly used methods for face alignment. CBR methods require less computer memory, though their performance is less than HBR methods. In this paper, we propose an Adaptive Coordinatebased Regression (ACR) loss to improve the accuracy of CBR for face alignment. Inspired by the Active Shape Model (ASM), we generate Smooth-Face objects, a set of facial landmark points with less variations compared to the ground truth landmark points. We then introduce a method to estimate the level of difficulty in predicting each landmark point for the network by comparing the distribution of the ground truth landmark points
and the corresponding Smooth-Face objects. Our proposed ACR Loss can adaptively modify its curvature and the influence of the loss based on the difficulty level of predicting each landmark point in a face. Accordingly, the ACR Loss guides the network toward challenging points than easier points, which improves the accuracy of the face alignment task. Our extensive evaluation shows the capabilities of the proposed ACR Loss in predicting facial landmark points in various facial images.

We evaluated our ACR Loss using MobileNetV2, EfficientNetB0, and EfficientNet-B3 on widely used 300W, and COFW datasets and showed that the performance of face alignment using the ACR Loss is much better than the widely-used L2 loss. Moreover, on the COFW dataset, we achieved state-of-theart accuracy. In addition, on 300W the ACR Loss performance is comparable to the state-of-the-art methods. We also compared the performance of MobileNetV2 trained using the ACR Loss with the lightweight state-of-the-art methods, and we achieved the best accuracy, highlighting the effectiveness of our ACR Loss for face alignment specifically for the lightweight models.
