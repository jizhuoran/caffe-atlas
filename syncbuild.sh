cd arm_build
make -j32
scp -P 1996 ~/caffe-atlas/arm_build/tools/caffe-d zrji@147.8.67.71:/home/zrji/caffe-atlas/build/tools/caffe
scp -P 1996 ~/caffe-atlas/arm_build/tools/caffe zrji@147.8.67.71:/home/zrji/caffe-atlas/build/tools/caffe


# scp -r -P 1996 /home/zrji/caffe-atlas-convertor/kernel_meta/ zrji@147.8.67.71:/home/zrji