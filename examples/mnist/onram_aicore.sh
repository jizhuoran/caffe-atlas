#!/usr/bin/env sh
set -e

./build/tools/caffe train --solver=examples_ram/mnist/onram_lenet_train_test.prototxt $@
