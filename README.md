This is a reimplementation of T-C3D(Temporal Convolutional 3D Network for Real-time Action Recognition) in caffe.

First of all, download the model here(https://pan.baidu.com/s/1jIS0iDg) and put it in /tc3d/examples/tc3d_ucf101_finetuning
Then, training the model by running the following commands:

cd tc3d
make -j
cd  examples/tc3d_ucf101_finetuning
sh finetuning_ucf101.sh
