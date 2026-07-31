#!/usr/bin/env bash
#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

cuda_include="${CUDA_INCLUDE:-/usr/local/cuda/include}"
build_dir="${BUILD_DIR:-$(mktemp -d)}"
trap 'rm -rf "${build_dir}"' EXIT
mkdir -p "${build_dir}/control"

cc -std=gnu11 -fPIC -shared -Wall -Wextra -Werror \
  -I"${cuda_include}" \
  -o "${build_dir}/libcuda.so.1" fake_cuda.c \
  -Wl,--version-script=fake_cuda.map -Wl,-Bsymbolic-functions
ln -s libcuda.so.1 "${build_dir}/libcuda.so"
cc -std=gnu11 -fPIC -shared -Wall -Wextra -Werror \
  -I"${cuda_include}" \
  -o "${build_dir}/libcudart.so.12" fake_cuda.c \
  -Wl,--version-script=fake_cuda.map -Wl,-Bsymbolic-functions
ln -s libcudart.so.12 "${build_dir}/libcudart.so"
cc -std=gnu11 -fPIC -shared -Wall -Wextra -Werror \
  -Wno-deprecated-declarations -I"${cuda_include}" \
  -o "${build_dir}/libdynamo_snapshot_cuda_vmm.so" ../interpose.c \
  -Wl,-Bsymbolic-functions -ldl -pthread
cc -std=gnu11 -Wall -Wextra -Werror -I"${cuda_include}" \
  -L"${build_dir}" -Wl,--no-as-needed -Wl,-rpath,'$ORIGIN' \
  -o "${build_dir}/interpose_test" interpose_test.c -l:libcuda.so.1
cc -std=gnu11 -Wall -Wextra -Werror -I"${cuda_include}" \
  -o "${build_dir}/explicit_local_test" explicit_local_test.c -ldl

(
  unset DYN_SNAPSHOT_CUDA_VMM_INTERPOSE
  LD_PRELOAD="${build_dir}/libdynamo_snapshot_cuda_vmm.so" \
    "${build_dir}/interpose_test" dormant
)
env \
  DYN_SNAPSHOT_CUDA_VMM_INTERPOSE=1 \
  DYN_SNAPSHOT_CONTROL_DIR="${build_dir}/control" \
  LD_PRELOAD="${build_dir}/libdynamo_snapshot_cuda_vmm.so" \
  "${build_dir}/interpose_test" resolvers
env \
  DYN_SNAPSHOT_CUDA_VMM_INTERPOSE=1 \
  DYN_SNAPSHOT_CONTROL_DIR="${build_dir}/control" \
  LD_LIBRARY_PATH="${build_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
  LD_PRELOAD="${build_dir}/libdynamo_snapshot_cuda_vmm.so" \
  "${build_dir}/explicit_local_test"
env \
  DYN_SNAPSHOT_CUDA_VMM_INTERPOSE=1 \
  DYN_SNAPSHOT_CONTROL_DIR="${build_dir}/control" \
  LD_PRELOAD="${build_dir}/libdynamo_snapshot_cuda_vmm.so" \
  "${build_dir}/interpose_test" explicit-loader
env \
  DYN_SNAPSHOT_CUDA_VMM_INTERPOSE=1 \
  DYN_SNAPSHOT_CONTROL_DIR="${build_dir}/control" \
  LD_PRELOAD="${build_dir}/libdynamo_snapshot_cuda_vmm.so" \
  "${build_dir}/interpose_test" peer-lifecycle
env \
  DYN_SNAPSHOT_CUDA_VMM_INTERPOSE=1 \
  DYN_SNAPSHOT_CONTROL_DIR="${build_dir}/control" \
  LD_PRELOAD="${build_dir}/libdynamo_snapshot_cuda_vmm.so" \
  "${build_dir}/interpose_test" release-failure-rollback
env \
  DYN_SNAPSHOT_CUDA_VMM_INTERPOSE=1 \
  DYN_SNAPSHOT_CONTROL_DIR="${build_dir}/control" \
  LD_PRELOAD="${build_dir}/libdynamo_snapshot_cuda_vmm.so" \
  "${build_dir}/interpose_test" multi-object-retry
env \
  DYN_SNAPSHOT_CUDA_VMM_INTERPOSE=1 \
  DYN_SNAPSHOT_CUDA_VMM_FORCE_POSIX=1 \
  DYN_SNAPSHOT_CONTROL_DIR="${build_dir}/control" \
  LD_PRELOAD="${build_dir}/libdynamo_snapshot_cuda_vmm.so" \
  "${build_dir}/interpose_test" force-posix-negotiation
env \
  DYN_SNAPSHOT_CUDA_VMM_INTERPOSE=1 \
  DYN_SNAPSHOT_CONTROL_DIR="${build_dir}/control" \
  LD_PRELOAD="${build_dir}/libdynamo_snapshot_cuda_vmm.so" \
  "${build_dir}/interpose_test" live-fabric
env \
  DYN_SNAPSHOT_CUDA_VMM_INTERPOSE=1 \
  DYN_SNAPSHOT_CUDA_VMM_FORCE_POSIX=1 \
  DYN_SNAPSHOT_CONTROL_DIR="${build_dir}/control" \
  LD_PRELOAD="${build_dir}/libdynamo_snapshot_cuda_vmm.so" \
  "${build_dir}/interpose_test" unsupported
env \
  DYN_SNAPSHOT_CUDA_VMM_INTERPOSE=1 \
  DYN_SNAPSHOT_CONTROL_DIR="${build_dir}/control" \
  LD_PRELOAD="${build_dir}/libdynamo_snapshot_cuda_vmm.so" \
  "${build_dir}/interpose_test" fork
env \
  DYN_SNAPSHOT_CUDA_VMM_INTERPOSE=1 \
  DYN_SNAPSHOT_CONTROL_DIR="${build_dir}/control" \
  LD_PRELOAD="${build_dir}/libdynamo_snapshot_cuda_vmm.so" \
  "${build_dir}/interpose_test" pre-cuda-fork
