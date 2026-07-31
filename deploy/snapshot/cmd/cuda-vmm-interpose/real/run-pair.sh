#!/usr/bin/env bash
#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

control_dir="${DYN_SNAPSHOT_CONTROL_DIR:-/snapshot-control}"
pair="${DYN_SNAPSHOT_CUDA_VMM_PAIR:-/usr/local/bin/dynamo-snapshot-cuda-vmm-pair}"
interposer="${DYN_SNAPSHOT_CUDA_VMM_LIBRARY:-/usr/local/lib/dynamo/libdynamo_snapshot_cuda_vmm.so}"
mkdir -p "${control_dir}"

env \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" \
  DYN_SNAPSHOT_CUDA_VMM_INTERPOSE=1 \
  DYN_SNAPSHOT_CUDA_VMM_FORCE_POSIX=1 \
  TORCH_SYMM_MEM_DISABLE_MULTICAST=1 \
  DYN_SNAPSHOT_CONTROL_DIR="${control_dir}" \
  "${pair}" "${interposer}"
