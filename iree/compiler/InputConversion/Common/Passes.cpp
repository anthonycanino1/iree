// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/InputConversion/Common/Passes.h"

#include "mlir/Pass/PassOptions.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {
namespace iree_compiler {

namespace {
#define GEN_PASS_REGISTRATION
#include "iree/compiler/InputConversion/Common/Passes.h.inc"  // IWYU pragma: export
}  // namespace

void registerCommonInputConversionPasses() {
  // Generated.
  registerPasses();
}

}  // namespace iree_compiler
}  // namespace mlir
