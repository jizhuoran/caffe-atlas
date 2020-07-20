#!/usr/bin/env sh
set -e

./on_build/tools/caffe train --solver=examples/mnist/lenet_aicore_solver.prototxt $@

