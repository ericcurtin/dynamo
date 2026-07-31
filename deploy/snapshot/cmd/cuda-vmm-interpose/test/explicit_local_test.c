/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#define _GNU_SOURCE

#include <cuda.h>
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>

static void
require(int condition, const char* message)
{
  if (!condition) {
    fprintf(stderr, "FAIL: %s\n", message);
    exit(1);
  }
}

int
main(void)
{
  typedef CUresult(CUDAAPI * create_type)(
      CUmemGenericAllocationHandle*, size_t, const CUmemAllocationProp*, unsigned long long);
  typedef CUresult(CUDAAPI * release_type)(CUmemGenericAllocationHandle);
  void* library = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
  create_type create;
  release_type release;
  CUmemAllocationProp properties = {0};
  CUmemGenericAllocationHandle handle;

  require(library != NULL, "RTLD_LOCAL libcuda load failed");
  create = (create_type)dlsym(library, "cuMemCreate");
  release = (release_type)dlsym(library, "cuMemRelease");
  require(create != NULL && release != NULL, "managed explicit symbols unavailable");
  require(create(&handle, 4096, &properties, 0) == CUDA_SUCCESS, "RTLD_LOCAL wrapper could not call the real driver");
  require(handle != 0, "RTLD_LOCAL native owner handle was not returned");
  require(release(handle) == CUDA_SUCCESS, "RTLD_LOCAL native owner handle release failed");
  dlclose(library);
  return 0;
}
