This is a reimplementation of T-C3D(Temporal Convolutional 3D Network for Real-time Action Recognition) in caffe.

First of all, download the model here(https://pan.baidu.com/s/1jIS0iDg or https://www.dropbox.com/s/631k99qv8sb8xll/tc3d_kinetics_pretrained.caffemodel?dl=0) 
and put it in /tc3d/examples/tc3d_ucf101_finetuning

Then, training the model by running the following commands:

git clone https://github.com/tc3d/tc3d.git

cd tc3d

make -j

cd  examples/tc3d_ucf101_finetuning

sh finetuning_ucf101.sh
