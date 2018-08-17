#!/bin/bash

set -ex

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
cd ${DIR} && cd ../onnx_xla && find . -iname '*.h' -o -iname '*.cc' | xargs clang-format -style=Chromium -i
cd ${DIR} && cd ../python_onnxifi && find . -iname '*.h' -o -iname '*.cc' | xargs clang-format -style=Chromium -i

