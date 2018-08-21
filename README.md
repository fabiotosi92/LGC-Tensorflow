# LGCNet-Tensorflow

Tensorflow implementation of a local-global framework for confidence estimation.

**Beyond local reasoning for stereo confidence estimation with deep learning**

[Fabio Tosi](https://vision.disi.unibo.it/~ftosi/), [Matteo Poggi](https://vision.disi.unibo.it/~mpoggi/), Antonio Benincasa and [Stefano Mattoccia](https://vision.disi.unibo.it/~smatt/Site/Home.html)   
ECCV 2018

## Qualitative results on KITTI
![Alt text](https://github.com/fabiotosi92/LGC-Tensorflow/blob/master/images/output.png "output")

Example of confidence estimation. (a) Reference image from KITTI 2015 dataset, (b) disparity map obtained with MC-CNN, (c) confidence estimated with a local approach (CCNN) and (d) the proposed local-global framework, highlighting regions on which the latter method provides more reliable predictions (red bounding boxes).

For more details: 
[pdf]


## Requirements
This code was tested with Tensorflow 1.4, CUDA 8.0 and Ubuntu 16.04.
