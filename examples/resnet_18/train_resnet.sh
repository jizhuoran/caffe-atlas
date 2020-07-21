#!/usr/bin/env sh
set -e

./build/tools/caffe-d train --solver=examples/resnet_18/solver.prototxt $@
