// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- TileAndVectorizeInOneWorkgroup.cpp ---------------------------------===//
//
// This pass tiles and vectorizes Linalg ops on buffers within in a single
// workgroup.
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Conversion/LinalgToSPIRV/KernelDispatchUtils.h"
#include "iree/compiler/Conversion/LinalgToSPIRV/MemorySpace.h"
#include "iree/compiler/Conversion/LinalgToSPIRV/Utils.h"
#include "iree/compiler/Conversion/PassDetail.h"
#include "iree/compiler/Conversion/Passes.h"
#include "iree/compiler/Conversion/Transforms/Transforms.h"
#include "iree/compiler/Conversion/Utils/MarkerUtils.h"
#include "iree/compiler/Conversion/Utils/TransformUtils.h"
#include "iree/compiler/Conversion/Utils/Utils.h"
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

//====---------------------------------------------------------------------===//
// Patterns for unrolling vectors
//====---------------------------------------------------------------------===//

static void populateVectorUnrollPatterns(MLIRContext *context,
                                         OwningRewritePatternList &patterns) {
  patterns.insert<vector::UnrollVectorPattern>(
      context,
      vector::UnrollVectorOptions().setNativeShapeFn(getSPIRVNativeVectorSize));
}

//====---------------------------------------------------------------------===//
// Vector patterns
//====---------------------------------------------------------------------===//

/*
static void applyVectorTransformation(FuncOp funcOp) {
  auto targetEnv = spirv::TargetEnv(spirv::lookupTargetEnv(funcOp));
  bool useCooperativeMatrix =
      targetEnv.allows(spirv::Capability::CooperativeMatrixNV) &&
      targetEnv.allows(spirv::Extension::SPV_NV_cooperative_matrix);
  {
    {
      OwningRewritePatternList vectorUnrollPatterns(funcOp.getContext());
      populateVectorUnrollPatterns(funcOp.getContext(), vectorUnrollPatterns);
      (void)applyPatternsAndFoldGreedily(funcOp,
                                         std::move(vectorUnrollPatterns));
    }
    {
      OwningRewritePatternList canonicalizationPatterns1(funcOp.getContext());

      vector::populateVectorToVectorTransformationPatterns(
          canonicalizationPatterns1);
      vector::populateVectorToVectorCanonicalizationPatterns(
          canonicalizationPatterns1);
      vector::populateSplitVectorTransferPatterns(canonicalizationPatterns1);
      (void)applyPatternsAndFoldGreedily(funcOp,
                                         std::move(canonicalizationPatterns1));

      OwningRewritePatternList canonicalizationPatterns2(funcOp.getContext());
      vector::populateVectorSlicesLoweringPatterns(canonicalizationPatterns2);
      vector::populateVectorTransferLoweringPatterns(canonicalizationPatterns2);
      (void)applyPatternsAndFoldGreedily(funcOp,
                                         std::move(canonicalizationPatterns2));

      if (useCooperativeMatrix) {
        // When using cooperative matrix we don't want to lower the contract,
        // instead we want to merge contract and transpose so that they can be
        // converted to cooperative matrix matmul op.
        // TODO(thomasraoux): remove that once we support cooperative matrix
        // lowering in MLIR core.
        OwningRewritePatternList combineTransposePatterns(funcOp.getContext());
        combineTransposePatterns.add<CombineContractTranspose>(
            funcOp.getContext());
        (void)applyPatternsAndFoldGreedily(funcOp,
                                           std::move(combineTransposePatterns));
      } else {
        OwningRewritePatternList contractLoweringPatterns(funcOp.getContext());
        vector::populateVectorContractLoweringPatterns(
            contractLoweringPatterns,
            vector::VectorTransformsOptions().setVectorTransformsOptions(
                vector::VectorContractLowering::OuterProduct));
        (void)applyPatternsAndFoldGreedily(funcOp,
                                           std::move(contractLoweringPatterns));
      }
    }
    LLVM_DEBUG({
      llvm::dbgs() << "--- After unrolling vector ---\n";
      funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
      llvm::dbgs() << "\n\n";
    });
  }

  {
    linalg::hoistRedundantVectorTransfers(funcOp);

    LLVM_DEBUG({
      llvm::dbgs() << "--- After hoisting vector transfers ---\n";
      funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
      llvm::dbgs() << "\n\n";
    });
  }
}
*/

//===----------------------------------------------------------------------===//
// Main pass
//===----------------------------------------------------------------------===//

namespace {
/// Function pass that implements tiling and fusion in Linalg on buffers.
class LinalgToVectorTileAndVectorize
    : public LinalgToVectorTileAndVectorizeBase<LinalgToVectorTileAndVectorize> {
 public:
  LinalgToVectorTileAndVectorize() {}
  LinalgToVectorTileAndVectorize(
      const LinalgToVectorTileAndVectorize &pass) {}

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

void LinalgToVectorTileAndVectorize::runOnOperation() {
  MLIRContext *context = &getContext();
  FuncOp funcOp = getOperation();

  SmallVector<linalg::LinalgOp, 4> linalgOps;
  SmallVector<Operation *, 4> tiledLoops;

  if (failed(getLinalgOps(funcOp, linalgOps, tiledLoops))) {
    // Nothing to do here.
    return;
  }

  llvm::dbgs() << "--- Before vectorization ---\n";
  funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
  llvm::dbgs() << "\n\n";

  promoteSingleIterationLoops(funcOp);

  OwningRewritePatternList vectorizationPatterns(&getContext());
  populateVectorizationPatterns(context, vectorizationPatterns);
  populateLinalgToVectorVectorizeConvPatterns(context, vectorizationPatterns);
  (void)applyPatternsAndFoldGreedily(funcOp, std::move(vectorizationPatterns));
  llvm::dbgs() << "--- After vectorization ---\n";
  funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
  llvm::dbgs() << "\n\n";

  /*
  // TODO: This should be a folding of Add into Contract in core but while
  // they live in different dialects, it is not possible without unnatural
  // dependencies.
  funcOp.walk([&](Operation *op) {
    if (auto contract = canonicalizeContractionAdd(op))
      op->replaceAllUsesWith(contract);
  });

  applyVectorTransformation(funcOp);
  */
}

//===----------------------------------------------------------------------===//
// Pass entry point and registration
//===----------------------------------------------------------------------===//

std::unique_ptr<OperationPass<FuncOp>>
createLinalgToVectorTileAndVectorize() {
  return std::make_unique<LinalgToVectorTileAndVectorize>();
}

}  // namespace iree_compiler
}  // namespace mlir
