// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_ROCM_BUFFER_H_
#define IREE_HAL_ROCM_BUFFER_H_

#include "experimental/rocm/rocm_headers.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Wraps a rocm allocation in an iree_hal_buffer_t.
iree_status_t iree_hal_rocm_buffer_wrap(
    iree_hal_allocator_t *allocator, iree_hal_memory_type_t memory_type,
    iree_hal_memory_access_t allowed_access,
    iree_hal_buffer_usage_t allowed_usage, iree_device_size_t allocation_size,
    iree_device_size_t byte_offset, iree_device_size_t byte_length,
    hipDeviceptr_t device_ptr, void *host_ptr, iree_hal_buffer_t **out_buffer);

// Returns the rocm base pointer for the given |buffer|.
// This is the entire allocated_buffer and must be offset by the buffer
// byte_offset and byte_length when used.
hipDeviceptr_t iree_hal_rocm_buffer_device_pointer(iree_hal_buffer_t *buffer);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_ROCM_BUFFER_H_
