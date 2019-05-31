# CRNN-with-STN
implement CRNN in Keras with Spatial Transformer Network (STN) for Optical Character Recognition(OCR)


The model is easy to start a trainning, but the performance of recognition is not better than the original CRNN without STN.



You can run CRNN individually by just remove the STN components, and connect *batchnorm_7* to *x_shape*. The CRNN can reach 90% of recognition accuracy.

Train on Synthetic Word Dataset realsed by M. Jaderberg et al. You can download the dataset [HERE](http://www.robots.ox.ac.uk/~vgg/data/text/#sec-synth)


## How to Run
Just run the CRNN_with_STN.py script.
My environment is **Tensorflow 1.4.0, Keras 2.0.9**. If you don't know how to install frameworks, please check [
Installing Deep Learning Frameworks on Ubuntu with CUDA support](https://www.learnopencv.com/installing-deep-learning-frameworks-on-ubuntu-with-cuda-support/).


## Reference
#### CRNN(Convolutional Recurrent Neural Network)
Shi, Baoguang, X. Bai, and C. Yao. ***"An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition."*** IEEE Transactions on Pattern Analysis & Machine Intelligence PP.99(2016):1-1.[[arxiv]](https://arxiv.org/abs/1507.05717)
#### STN(Spatial Transformer Network)
Max Jaderberg, Karen Simonyan,Andrew Zisserman and Koray Kavukcuoglu. ***"Spatial Transformer Network"*** [[arxiv]](https://arxiv.org/abs/1506.02025)

## 2019.03.13 UPDATE
I have tried to put STN part in the front of the network, between batchnorm_1 and conv_2, than the network didn't converge at all. Maybe I used STN in a wrong way?

## Weights Download, 2019.05.31 UPDATE
#### 1. ou can download trained weights [CRNN_without_STN](https://drive.google.com/file/d/13rcnDxRiDBDKgg-1mwikdc7F1SYG1y-g/view?usp=sharing), and [CRNN_with_STN](https://drive.google.com/file/d/1n1Vlsz77SBh_b2cviC464ECrRhD-GSVr/view?usp=sharing). (Google Drive)

#### 2. If you can't access google, here is the Baidu Netdisk [share link](https://pan.baidu.com/s/1vHOC5Ba9mYk4lEkHRMbd5A), with password "89vw".

#### 3. Added a DEMO for model predicting
