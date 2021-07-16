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
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
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

//
LogicalResult convertBinaryArithOps(FuncOp funcOp) {
  SmallVector<AddIOp, 4> vectorAddOps;

  funcOp.getOperation()->walk([&](AddIOp op) {
    if (!op.lhs().getType().template isa<VectorType>()) {
      return;
    }

    auto resultVectorType =
        op->getResult(0).getType().template dyn_cast<VectorType>();
    // Only work on rank 1 vectors for now

    if (resultVectorType.getRank() != 1) return;

    auto lhsReadOp = op.lhs().getDefiningOp<vector::TransferReadOp>();
    auto rhsReadOp = op.rhs().getDefiningOp<vector::TransferReadOp>();

    Operation *user = *op.result().getUsers().begin();
    auto dstWriteOp = dyn_cast<vector::TransferWriteOp>(user);

    if (!lhsReadOp || !rhsReadOp || !dstWriteOp) return;

    vectorAddOps.push_back(op);
  });

  for (auto op : vectorAddOps) {
    OpBuilder b(op);

    auto resultVectorType =
        op->getResult(0).getType().template dyn_cast<VectorType>();

    auto lhsReadOp = op.lhs().getDefiningOp<vector::TransferReadOp>();
    auto rhsReadOp = op.rhs().getDefiningOp<vector::TransferReadOp>();

    Operation *user = *op.result().getUsers().begin();
    auto dstWriteOp = dyn_cast<vector::TransferWriteOp>(user);

    auto lengthOp = b.create<ConstantIndexOp>(
        op.getLoc(), resultVectorType.getNumElements());

    switch (resultVectorType.getElementTypeBitWidth()) {
      case 32: {
        b.create<IREE::VMVX::AddSI32Op>(
            op.getLoc(), lhsReadOp.source(), *lhsReadOp.indices().begin(),
            rhsReadOp.source(), *rhsReadOp.indices().begin(),
            dstWriteOp.source(), *dstWriteOp.indices().begin(), lengthOp);
        break;
      }
      case 64: {
        b.create<IREE::VMVX::AddSI64Op>(
            op.getLoc(), lhsReadOp.source(), *lhsReadOp.indices().begin(),
            rhsReadOp.source(), *rhsReadOp.indices().begin(),
            dstWriteOp.source(), *dstWriteOp.indices().begin(), lengthOp);
        break;
      }
      default:
        return failure();
    }

    dstWriteOp.erase();
    op.erase();
    lhsReadOp.erase();
    rhsReadOp.erase();
  }

  return success();
}

void populateVectorToVMVXPatterns(MLIRContext *context,
                                  OwningRewritePatternList &patterns,
                                  TypeConverter &typeConverter) {}

}  // namespace iree_compiler
}  // namespace mlir
