// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- VMVXTileAndVectorize.cpp ---------------------------------===//
//
// This pass tiles and vectorizes Linalg ops on buffers for downstream
// VMVX conversion.
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Codegen/SPIRV/KernelDispatchUtils.h"
#include "iree/compiler/Codegen/SPIRV/MemorySpace.h"
#include "iree/compiler/Codegen/SPIRV/Utils.h"
#include "iree/compiler/Codegen/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Utils/MarkerUtils.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/Shape/IR/ShapeDialect.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/Linalg/Analysis/DependenceAnalysis.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/Transforms/Hoisting.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Vector/VectorTransforms.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Identifier.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/FoldUtils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/LoopUtils.h"

namespace mlir {
namespace iree_compiler {

//===----------------------------------------------------------------------===//
// Utility functions
//===----------------------------------------------------------------------===//

/// Returns a Linalg marker that matches any of the `matchMarkers` and replaces
/// it with `replaceMarker`.
static linalg::LinalgTransformationFilter getLinalgMatchAndReplaceMarker(
    ArrayRef<StringRef> matchMarkers, StringRef replaceMarker,
    MLIRContext *context) {
  SmallVector<Identifier, 2> markers;
  markers.reserve(matchMarkers.size());
  for (StringRef marker : matchMarkers) {
    markers.emplace_back(Identifier::get(marker, context));
  }
  return linalg::LinalgTransformationFilter(
      markers, Identifier::get(replaceMarker, context));
}

/// Converts a symbolic GPU processor dimension to its numeric one.
static unsigned dimToIndex(StringRef dim) {
  return StringSwitch<unsigned>(dim).Case("x", 0).Case("y", 1).Case("z", 2);
}

//====---------------------------------------------------------------------===//
// Patterns for vectorization
//====---------------------------------------------------------------------===//

static void populateVectorizationPatterns(MLIRContext *context,
                                          OwningRewritePatternList &patterns) {
  linalg::insertVectorizationPatterns<linalg::FillOp, linalg::GenericOp,
                                      linalg::ContractionOpInterface>(
      patterns, linalg::LinalgVectorizationOptions(),
      linalg::LinalgTransformationFilter(getLinalgMatchAndReplaceMarker(
          {getWorkgroupMarker(), getVectorizeMarker()}, getVectorizeMarker(),
          context)));
}

//===----------------------------------------------------------------------===//
// Main pass
//===----------------------------------------------------------------------===//

namespace {
/// Function pass that implements tiling and fusion in Linalg on buffers.
class VMVXTileAndVectorize
    : public VMVXTileAndVectorizeBase<VMVXTileAndVectorize> {
 public:
  VMVXTileAndVectorize() {}
  VMVXTileAndVectorize(const VMVXTileAndVectorize &pass) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AffineDialect, IREE::HAL::HALDialect, gpu::GPUDialect,
                    linalg::LinalgDialect, scf::SCFDialect, ShapeDialect,
                    vector::VectorDialect>();
  }

  void runOnOperation() override;
};
}  // namespace

//====---------------------------------------------------------------------===//
// Main pass implementation
//====---------------------------------------------------------------------===//

void VMVXTileAndVectorize::runOnOperation() {
  MLIRContext *context = &getContext();
  FuncOp funcOp = getOperation();

  SmallVector<linalg::LinalgOp, 4> linalgOps;
  SmallVector<Operation *, 4> tiledLoops;

  if (failed(getLinalgOps(funcOp, linalgOps, tiledLoops))) {
    // Nothing to do here.
    return;
  }

  OwningRewritePatternList vectorizationPatterns(&getContext());
  populateVectorizationPatterns(context, vectorizationPatterns);
  (void)applyPatternsAndFoldGreedily(funcOp, std::move(vectorizationPatterns));
}

//===----------------------------------------------------------------------===//
// Pass entry point and registration
//===----------------------------------------------------------------------===//

std::unique_ptr<OperationPass<FuncOp>> createVMVXTileAndVectorize() {
  return std::make_unique<VMVXTileAndVectorize>();
}

}  // namespace iree_compiler
}  // namespace mlir
