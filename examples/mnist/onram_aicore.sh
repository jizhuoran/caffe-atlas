#!/usr/bin/env sh
set -e

./build/tools/caffe train --solver=examples_ram/mnist/onram_lenet_aicore_solver.prototxt $@
