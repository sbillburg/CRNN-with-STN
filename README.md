# CRNN-with-STN
implement CRNN in Keras with Spatial Transformer  Network


Trainable but performance of recognition is not improved, meanwhile the loss raises up.

You can run CRNN individually by just remove the STN components as well. The CRNN can reach 90% of recognition accuracy.

Train on Synthetic Word Dataset realsed by M. Jaderberg et al. You can download the dataset [HERE](http://www.robots.ox.ac.uk/~vgg/data/text/#sec-synth)


## Reference

### CRNN(Convolutional Recurrent Neural Network)
Shi, Baoguang, X. Bai, and C. Yao. ***"An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition."*** IEEE Transactions on Pattern Analysis & Machine Intelligence PP.99(2016):1-1.[[arxiv]](https://arxiv.org/abs/1507.05717)

### STN(Spatial Transformer Network)
Max Jaderberg, Karen Simonyan,Andrew Zisserman and Koray Kavukcuoglu. ***"Spatial Transformer Network"*** [[arxiv]](https://arxiv.org/abs/1506.02025)

