#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Before doing anything, cd to the directory containing this script.
cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null || /bin/true

# Disable CoreML backend build by default. Users can override by setting CMAKE_ARGS.
if [[ -z "${CMAKE_ARGS:-}" ]] || [[ "${CMAKE_ARGS}" != *"-DEXECUTORCH_BUILD_COREML="* ]]; then
  export CMAKE_ARGS="${CMAKE_ARGS:-} -DEXECUTORCH_BUILD_COREML=OFF"
fi

./run_python_script.sh ./install_executorch.py "$@"
