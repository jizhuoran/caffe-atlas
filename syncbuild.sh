cd arm_build
make -j32
scp -P 1996 ~/ref-caffe/arm_build/tools/caffe-d zrji@147.8.67.71:/home/zrji/caffe-atlas/build/tools/caffe
scp -P 1996 ~/ref-caffe/arm_build/tools/caffe zrji@147.8.67.71:/home/zrji/caffe-atlas/build/tools/caffe