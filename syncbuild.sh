cd build
make -j32
# scp ~/caffe-atlas/build/tools/caffe-d zrji@192.168.0.133:/home/zrji/caffe-atlas/build/tools/caffe
# scp ~/caffe-atlas/build/tools/caffe zrji@192.168.0.133:/home/zrji/caffe-atlas/build/tools/caffe
# scp -r ~/ascend_generator/kernel_meta zrji@192.168.0.133:/home/zrji/caffe-atlas

scp -P 1996 ~/caffe-atlas/build/tools/caffe-d zrji@147.8.67.71:/home/zrji/caffe-atlas/build/tools/caffe
scp -P 1996 ~/caffe-atlas/build/tools/caffe zrji@147.8.67.71:/home/zrji/caffe-atlas/build/tools/caffe
# scp -P 1996 -r ~/ascend_generator/kernel_meta zrji@147.8.67.71:/home/zrji/caffe-atlas

# scp -P 10001 ~/caffe-atlas/build/tools/caffe-d zrji@147.8.177.155:/home/zrji/caffe-atlas/build/tools/caffe
# scp -P 10001 ~/caffe-atlas/build/tools/caffe zrji@147.8.177.155:/home/zrji/caffe-atlas/build/tools/caffe
# scp -P 10001 -r ~/ascend_generator/kernel_meta zrji@147.8.177.155:/home/zrji/caffe-atlas
