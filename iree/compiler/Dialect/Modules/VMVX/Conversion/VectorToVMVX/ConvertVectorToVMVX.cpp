// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Modules/VMVX/Conversion/VectorToVMVX/ConvertVectorToVMVX.h"

#include "iree/compiler/Dialect/IREE/IR/IREETypes.h"
#include "iree/compiler/Dialect/Modules/VMVX/IR/VMVXDialect.h"
#include "iree/compiler/Dialect/Modules/VMVX/IR/VMVXOps.h"
#include "iree/compiler/Dialect/Modules/VMVX/IR/VMVXTypes.h"
#include "iree/compiler/Dialect/Shape/IR/Builders.h"
#include "iree/compiler/Dialect/Shape/IR/ShapeOps.h"
#include "iree/compiler/Dialect/Shape/IR/ShapeTypes.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

#define DEBUG_TYPE "iree-convert-vector-to-vmvx"

namespace mlir {
namespace iree_compiler {

namespace {
class VectorReadConversion
    : public OpConversionPattern<vector::TransferReadOp> {
 public:
  using OpConversionPattern<vector::TransferReadOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      vector::TransferReadOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    llvm::dbgs() << "[A] Looking at " << op << " for conversion\n";
    return failure();
  }
};

template <typename SourceOp, typename Dst32Op, typename Dst64Op>
class BinaryArithmeticOpConversion : public OpConversionPattern<SourceOp> {
 public:
  using OpConversionPattern<SourceOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      SourceOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    llvm::dbgs() << "[A] Looking at " << op << " for conversion\n";

    typename SourceOp::Adaptor srcAdaptor(operands);
    if (!srcAdaptor.lhs().getType().template isa<VectorType>()) {
      return failure();
    }

    auto resultVectorType =
        op->getResult(0).getType().template dyn_cast<VectorType>();
    // Only work on rank 1 vectors for now

    llvm::dbgs() << "[A] Checking " << resultVectorType << "\n";

    if (resultVectorType.getRank() != 1) return failure();

    auto lhsReadOp =
        srcAdaptor.lhs().template getDefiningOp<vector::TransferReadOp>();
    auto rhsReadOp =
        srcAdaptor.rhs().template getDefiningOp<vector::TransferReadOp>();

    Operation *user = *op.result().getUsers().begin();
    auto dstWriteOp = dyn_cast<vector::TransferWriteOp>(user);

    llvm::dbgs() << "[A] lhsReadOp:" << lhsReadOp << "\n";
    llvm::dbgs() << "[A] rhsReadOp:" << rhsReadOp << "\n";
    llvm::dbgs() << "[A] dstWriteOp:" << dstWriteOp << "\n";

    if (!lhsReadOp || !rhsReadOp || !dstWriteOp) return failure();

    rewriter.setInsertionPointToStart(op.getOperation()->getBlock());
    auto lengthOp = rewriter.create<ConstantIndexOp>(
        op.getLoc(), resultVectorType.getNumElements());

    llvm::dbgs() << "[A] created op " << lengthOp << "\n";

    switch (resultVectorType.getElementTypeBitWidth()) {
      case 32:
        rewriter.replaceOpWithNewOp<Dst32Op>(
            op, lhsReadOp.source(), *lhsReadOp.indices().begin(),
            rhsReadOp.source(), *rhsReadOp.indices().begin(),
            dstWriteOp.source(), *dstWriteOp.indices().begin(), lengthOp);
        break;
      case 64:
        rewriter.replaceOpWithNewOp<Dst64Op>(
            op, lhsReadOp.source(), *lhsReadOp.indices().begin(),
            rhsReadOp.source(), *rhsReadOp.indices().begin(),
            dstWriteOp.source(), *dstWriteOp.indices().begin(), lengthOp);
        break;
      default:
        return rewriter.notifyMatchFailure(op, "unsupported type");
    }

    lhsReadOp.erase();
    rhsReadOp.erase();
    dstWriteOp.erase();

    return success();
  }
};

}  // namespace

void populateVectorToVMVXPatterns(MLIRContext *context,
                                  OwningRewritePatternList &patterns,
                                  TypeConverter &typeConverter) {
  patterns.insert<VectorReadConversion,
                  BinaryArithmeticOpConversion<AddIOp, IREE::VMVX::AddSI32Op,
                                               IREE::VMVX::AddSI64Op>>(
      typeConverter, context);
}

}  // namespace iree_compiler
}  // namespace mlir
