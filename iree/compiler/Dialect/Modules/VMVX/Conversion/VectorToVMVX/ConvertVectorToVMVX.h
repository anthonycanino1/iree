// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_VMVX_CONVERSION_VECTORTOVMVX_CONVERTVECTORTOVMVX_H_
#define IREE_COMPILER_DIALECT_VMVX_CONVERSION_VECTORTOVMVX_CONVERTVECTORTOVMVX_H_

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {

// Populates conversion patterns from the std dialect to the VMVX dialect.
void populateVectorToVMVXPatterns(MLIRContext *context,
                                    OwningRewritePatternList &patterns,
                                    TypeConverter &typeConverter);

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_VMVX_CONVERSION_VECTORTOVMVX_CONVERTVECTORTOVMVX_H_
        // // NOLINT
