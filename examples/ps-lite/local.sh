#!/bin/sh

sudo pkill -9 caffe
sudo pkill -9 ps_lite_server

if [ $# -lt 3 ]; then
    echo "usage: $0 num_servers num_workers bin [args..]"
    exit -1;
fi

export DMLC_NUM_SERVER=$1
shift
export DMLC_NUM_WORKER=$1
shift
bin=$1
shift
arg="$@"

echo ${DMLC_NUM_SERVER}
echo ${DMLC_NUM_WORKER}
echo ${bin}

# start the scheduler
export DMLC_PS_ROOT_URI='127.0.0.1'
export DMLC_PS_ROOT_PORT=8000
export DMLC_ROLE='scheduler'
/home/zrji/caffe-atlas/build/tools/ps_lite_server &


# start servers
export DMLC_ROLE='server'
export HEAPPROFILE=./S0
/home/zrji/caffe-atlas/build/tools/ps_lite_server &

echo "Going to start workers"

# start workers
export DMLC_ROLE='worker'
export HEAPPROFILE=./W0
/home/zrji/caffe-atlas/build/tools/caffe train --solver=examples/resnet_18/solver.prototxt &

# echo "Going to start workers 2"

# export HEAPPROFILE=./W1
# /home/zrji/caffe-atlas/build/tools/caffe train --solver=examples/resnet_18/solver.prototxt &

# export HEAPPROFILE=./W2
# /home/zrji/caffe-atlas/build/tools/caffe train --solver=examples/resnet_18/solver.prototxt &

wait
