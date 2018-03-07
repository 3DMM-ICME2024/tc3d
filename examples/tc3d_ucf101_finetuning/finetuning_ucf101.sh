mkdir -p LOG_TC3D_UCF101_split1
GLOG_log_dir="./LOG_TC3D_UCF101_split1" ../../.build_release/tools/caffe.bin train --solver=solver_r2.prototxt --weights=./tc3d_kinetics_pretrained.caffemodel --gpu=0,1
