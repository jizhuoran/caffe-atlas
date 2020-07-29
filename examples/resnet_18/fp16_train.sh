#!/usr/bin/env sh
set -e

./on_build/tools/caffe train --solver=examples/resnet_18/solver.prototxt $@
