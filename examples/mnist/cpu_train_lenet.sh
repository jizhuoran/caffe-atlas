#!/usr/bin/env sh
set -e

./build/tools/caffe train --solver=examples/mnist/cpu_lenet_solver.prototxt $@
