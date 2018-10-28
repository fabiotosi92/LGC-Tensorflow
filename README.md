# LGCNet-Tensorflow

Tensorflow implementation of a local-global framework for confidence estimation.

**Beyond local reasoning for stereo confidence estimation with deep learning**

[Fabio Tosi](https://vision.disi.unibo.it/~ftosi/), [Matteo Poggi](https://vision.disi.unibo.it/~mpoggi/), Antonio Benincasa and [Stefano Mattoccia](https://vision.disi.unibo.it/~smatt/Site/Home.html)   
ECCV 2018

## Qualitative results on KITTI
![Alt text](https://github.com/fabiotosi92/LGC-Tensorflow/blob/master/images/output.png "output")

Example of confidence estimation. (a) Reference image from KITTI 2015 dataset, (b) disparity map obtained with MC-CNN, (c) confidence estimated with a local approach (CCNN) and (d) the proposed local-global framework, highlighting regions on which the latter method provides more reliable predictions (red bounding boxes).

For more details: 
[pdf](https://vision.disi.unibo.it/~ftosi/papers/eccv18_lgc.pdf)

## Training

For training, the KITTI input images have been padded to 384x1280. 

The __training file__ should be a _.txt_ file in which each line contains: [_path_image_left_] [_path_image_disparity_] [_path_image_groundtruth_]. Disparity maps and groundtruth as 16 bit images.

You can train the __local__ network as follows: 

```shell     
python ./model/main.py --is_training True --epoch 14 --batch_size 128 --patch_size 9 --dataset [path_training_file] --initial_learning_rate 0.003 --log_directory [path_log] --save_epoch_freq 2 --model_name model --model [CCNN, EFN, LFN] 
```
Use the _--model_ argument to choose the architecture:
  * [__CCNN__](https://github.com/fabiotosi92/CCNN-Tensorflow/edit/master/README.md) ([Poggi et al.](https://vision.disi.unibo.it/~mpoggi/papers/bmvc2016.pdf))
  * __EFN__ (Early Fusion Network)
  * __LFN__ (Late Fusion Network)
          
Similarly, you can train the __global__ network (_ConfNet_):
```shell     
python ./model/main.py --is_training True --epoch 1600 --batch_size 1 --crop_height 256 --crop_width 512 --dataset [path_training_file] --initial_learning_rate 0.003 --log_directory [path_log] --model_name model --model ConfNet 
```
Finally, you can load weights from the local and global networks and train __LGCNet__ thereafter:

```shell 
python ./model/main.py --is_training True --epoch 14 --batch_size 128 --patch_size 9 --dataset [path_training_file] --initial_learning_rate 0.003 --log_directory [path_log] --save_epoch_freq 2 --model_name model --model LGC --checkpoint_path [path_checkpoint_ConfNet] [path_checkpoint_CCNN/LFN] --late_fusion
```
Use _--late_fusion_ flag to set __LFN__ as local network.

**Warning:** set _checkpoint_CCNN/LFN_ accordingly for the late fusion network.

## Testing

The __testing file__ should be a _.txt_ file in which each line contains: [_path_image_left_] [_path_image_disparity_]. Disparity maps as 16 bit images.

If you want to test the local network or the global network indipendently:

```shell
python ./model/main.py --is_training False --batch_size 1 --dataset [path_testing_file] --checkpoint_path [path_checkpoint] --output_path [path_output] --model [CCNN, EFN, LFN, ConfNet]
```

For testing LGCNet, instead, you can run:

```shell
python ./model/main.py --is_training False --batch_size 1 --dataset [path_testing_file] --initial_learning_rate 0.003 --model LGC --checkpoint_path [path_checkpoint_ConfNet] [path_checkpoint_CCNN/LFN] [path_checkpoint_LGC] --output_path [path_output] --late_fusion
```

## Pretrained models

You can download the pre-trained models, trained on 20 images of KITTI 12 dataset, here:

[Google Drive](https://drive.google.com/open?id=1gXThUY_6pRG2HozAyMB_tY4urd0rIPZh)

## Requirements
This code was tested with Tensorflow 1.4, CUDA 8.0 and Ubuntu 16.04.
