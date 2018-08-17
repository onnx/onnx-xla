#!/bin/bash

set -ex

# realpath might not be available on MacOS
script_path=$(python -c "import os; import sys; print(os.path.realpath(sys.argv[1]))" "${BASH_SOURCE[0]}")
top_dir=$(dirname $(dirname "$script_path"))
tf_dir="$top_dir/third_party/tensorflow"
build_dir="$top_dir/build/xla"
mkdir -p "$build_dir"

cd "$tf_dir"
git reset --hard
git clean -f
patch -p1 < $top_dir/tensorflow.patch

bazel build -c opt //tensorflow/compiler/tf2xla/lib:util
bazel build -c opt //tensorflow/compiler/xla/rpc:libxla_computation_client.so
bazel build -c opt //tensorflow/compiler/xla/rpc:grpc_service_main_cpu
