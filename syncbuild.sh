cd build
make -j32
scp ~/caffe-atlas/build/tools/caffe-d zrji@192.168.0.133:/home/zrji/caffe-ascend/build/tools/caffe
scp -r ~/ascend_generator/kernel_meta zrji@192.168.0.133:/home/zrji/caffe-ascend
