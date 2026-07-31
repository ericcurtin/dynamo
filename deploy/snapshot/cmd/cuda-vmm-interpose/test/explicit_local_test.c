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
#include <string.h>

CUresult CUDAAPI cuMemCreate(CUmemGenericAllocationHandle*, size_t, const CUmemAllocationProp*, unsigned long long)
    __attribute__((weak));
CUresult CUDAAPI cuMemRelease(CUmemGenericAllocationHandle) __attribute__((weak));

static void
require(int condition, const char* message)
{
  if (!condition) {
    fprintf(stderr, "FAIL: %s\n", message);
    exit(1);
  }
}

static void
test_explicit_local(int dormant, int versioned)
{
  typedef CUresult(CUDAAPI * create_type)(
      CUmemGenericAllocationHandle*, size_t, const CUmemAllocationProp*, unsigned long long);
  typedef void* (*real_create_type)(void);
  void* library = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
  create_type create;
  real_create_type real_create;
  CUmemAllocationProp properties = {0};
  CUmemGenericAllocationHandle handle;

  require(library != NULL, "RTLD_LOCAL libcuda load failed");
  real_create = (real_create_type)dlsym(library, "fake_cuda_real_mem_create");
  create = versioned ? (create_type)dlvsym(library, "cuMemCreate", "FAKE_CUDA_1.0")
                     : (create_type)dlsym(library, "cuMemCreate");
  require(create != NULL && real_create != NULL, "explicit fake CUDA symbols unavailable");
  require(
      dormant ? (void*)create == real_create() : (void*)create == (void*)&cuMemCreate,
      dormant ? "dormant explicit dlsym did not return the exact real symbol"
              : "enabled explicit dlsym did not return the wrapper");
  dlclose(library);
  require(
      (dormant ? cuMemCreate(&handle, 4096, &properties, 0) : create(&handle, 4096, &properties, 0)) == CUDA_SUCCESS,
      "direct wrapper could not forward through retained RTLD_LOCAL libcuda");
  require(handle != 0, "direct wrapper did not return a native owner handle");
  require(cuMemRelease(handle) == CUDA_SUCCESS, "direct wrapper could not release the native owner handle");
}

static void
test_lookup_only(int versioned)
{
  typedef CUresult(CUDAAPI * create_type)(
      CUmemGenericAllocationHandle*, size_t, const CUmemAllocationProp*, unsigned long long);
  typedef void* (*real_create_type)(void);
  void* library = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
  create_type create;
  real_create_type real_create;
  void* real;
  void* loaded;

  require(library != NULL, "RTLD_LOCAL libcuda load failed");
  real_create = (real_create_type)dlsym(library, "fake_cuda_real_mem_create");
  require(real_create != NULL, "fake CUDA real-symbol helper unavailable");
  real = real_create();
  create = versioned ? (create_type)dlvsym(library, "cuMemCreate", "FAKE_CUDA_1.0")
                     : (create_type)dlsym(library, "cuMemCreate");
  require(create != NULL && (void*)create == real, "dormant lookup-only request did not return the exact real symbol");
  dlclose(library);
  loaded = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL | RTLD_NOLOAD);
  if (loaded != NULL)
    dlclose(loaded);
  require(loaded == NULL, "dormant lookup-only request retained libcuda");
}

static void
test_first_direct(void)
{
  CUmemAllocationProp properties = {0};
  CUmemGenericAllocationHandle handle;
  void* loaded = dlopen("libcuda.so.1", RTLD_NOW | RTLD_NOLOAD);

  require(loaded == NULL, "interposer constructor eagerly loaded libcuda");
  require(cuMemCreate(&handle, 4096, &properties, 0) == CUDA_SUCCESS, "first direct wrapper did not load libcuda");
  require(handle != 0, "first direct wrapper did not return a native owner handle");
  require(cuMemRelease(handle) == CUDA_SUCCESS, "first direct wrapper could not release the native owner handle");
}

int
main(int argc, char** argv)
{
  require(argc == 2, "one test mode is required");
  if (strcmp(argv[1], "explicit-local") == 0)
    test_explicit_local(1, 0);
  else if (strcmp(argv[1], "explicit-local-versioned") == 0)
    test_explicit_local(1, 1);
  else if (strcmp(argv[1], "enabled-explicit-local") == 0)
    test_explicit_local(0, 0);
  else if (strcmp(argv[1], "lookup-only") == 0)
    test_lookup_only(0);
  else if (strcmp(argv[1], "lookup-only-versioned") == 0)
    test_lookup_only(1);
  else if (strcmp(argv[1], "first-direct") == 0)
    test_first_direct();
  else
    require(0, "unknown test mode");
  return 0;
}
