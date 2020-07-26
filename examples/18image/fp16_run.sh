#!/usr/bin/env sh
set -e

./on_build/tools/caffe train --solver=examples/18image/resnet_solver.prototxt $@