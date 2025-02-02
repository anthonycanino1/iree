// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"

#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/SMLoc.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
namespace iree_compiler {
namespace linalg_ext {

//===----------------------------------------------------------------------===//
// Utils.
//===----------------------------------------------------------------------===//

static void getEffectsImpl(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects,
    ValueRange results, ValueRange inputBuffers, ValueRange outputBuffers) {
  for (Value value : results) {
    effects.emplace_back(MemoryEffects::Allocate::get(), value,
                         SideEffects::DefaultResource::get());
  }
  for (Value value : inputBuffers) {
    effects.emplace_back(MemoryEffects::Read::get(), value,
                         SideEffects::DefaultResource::get());
  }
  for (Value value : outputBuffers) {
    effects.emplace_back(MemoryEffects::Read::get(), value,
                         SideEffects::DefaultResource::get());
    effects.emplace_back(MemoryEffects::Write::get(), value,
                         SideEffects::DefaultResource::get());
  }
}

//===----------------------------------------------------------------------===//
// Common methods from Linalg dialect.
//===----------------------------------------------------------------------===//

static ParseResult parseLinalgExtOperandList(
    OpAsmParser &parser, StringRef keyword,
    SmallVectorImpl<OpAsmParser::OperandType> &values,
    SmallVectorImpl<Type> &types) {
  StringRef parsedKeyword;
  if (succeeded(parser.parseOptionalKeyword(&parsedKeyword, {keyword}))) {
    if (parser.parseLParen() || parser.parseOperandList(values) ||
        parser.parseColonTypeList(types) || parser.parseRParen()) {
      return failure();
    }
  }
  return success();
}

static ParseResult parseLinalgExtInsList(
    OpAsmParser &parser, SmallVectorImpl<OpAsmParser::OperandType> &values,
    SmallVectorImpl<Type> &types) {
  return parseLinalgExtOperandList(parser, "ins", values, types);
}

static ParseResult parseLinalgExtOutsList(
    OpAsmParser &parser, SmallVectorImpl<OpAsmParser::OperandType> &values,
    SmallVectorImpl<Type> &types) {
  return parseLinalgExtOperandList(parser, "outs", values, types);
}

static void printLinalgExtOperandList(OpAsmPrinter &printer, Operation *op,
                                      StringRef keyword, OperandRange values,
                                      TypeRange types) {
  if (!values.empty()) {
    printer << keyword << "(";
    printer.printOperands(values);
    printer << " : " << types << ")";
  }
}

static void printLinalgExtInsList(OpAsmPrinter &printer, Operation *op,
                                  OperandRange values, TypeRange types) {
  return printLinalgExtOperandList(printer, op, "ins", values, types);
}

static void printLinalgExtOutsList(OpAsmPrinter &printer, Operation *op,
                                   OperandRange values, TypeRange types) {
  return printLinalgExtOperandList(printer, op, "outs", values, types);
}

//===----------------------------------------------------------------------===//
// ScatterOp
//===----------------------------------------------------------------------===//

void ScatterOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  SmallVector<Value> inputBuffers = getInputBufferOperands();
  SmallVector<Value> outputBuffers = getOutputBufferOperands();
  getEffectsImpl(effects, getOperation()->getResults(), inputBuffers,
                 outputBuffers);
}

static LogicalResult verifyScatterOp(ScatterOp op) {
  if (op.inputs().size() != 2) {
    return op.emitOpError("expected two input operands");
  }
  if (op.outputs().size() != 1) {
    return op.emitOpError("expected one output operand");
  }
  auto checkDimensionsMatch = [&](ShapedType t1, ShapedType t2, unsigned dim) {
    return t1.getShape()[dim] == t2.getShape()[dim];
  };
  auto indicesType = op.inputs()[1].getType().cast<ShapedType>();
  if (indicesType.getRank() != 1 ||
      !indicesType.getElementType().isInteger(32)) {
    return op.emitOpError(
        "expected indices to be of rank 1 of i32 element type");
  }
  // The first dimension of the indices should match the first dimension of the
  // output.
  auto updateType = op.inputs()[0].getType().cast<ShapedType>();
  if (updateType.getRank() < 1) {
    return op.emitOpError("expected update value to be at least rank 1");
  }
  if (!checkDimensionsMatch(indicesType, updateType, 0)) {
    return op.emitOpError(
        "mismatch in shape of indices and update value at dim#0");
  }
  auto originalType = op.outputs()[0].getType().cast<ShapedType>();
  if (originalType.getRank() != updateType.getRank()) {
    return op.emitOpError(
        "mismatch in rank of update value and original value");
  }
  for (auto dim : llvm::seq<unsigned>(1, originalType.getRank())) {
    if (!checkDimensionsMatch(updateType, originalType, dim)) {
      return op.emitOpError(
                 "mismatch in shape of update value and original value at dim#")
             << dim;
    }
  }
  Region &region = op.region();
  Block *body = &region.front();
  if (body->getNumArguments() != 2) {
    return op.emitOpError("expected region to have two arguments");
  }
  Type arg0Type = body->getArgument(0).getType();
  Type arg1Type = body->getArgument(1).getType();
  if (!arg0Type.isIntOrFloat() || !arg1Type.isIntOrFloat()) {
    return op.emitOpError(
        "expected region to have scalar argument of integer or float types");
  }
  if (arg0Type != updateType.getElementType()) {
    return op.emitOpError("mismatch in argument 0 of region ")
           << arg0Type << " and element type of update value "
           << updateType.getElementType();
  }
  if (arg1Type != originalType.getElementType()) {
    return op.emitOpError("mismatch in argument 1 of region ")
           << arg1Type << " and element type of original value "
           << originalType.getElementType();
  }
  if (arg0Type != arg1Type) {
    return op.emitOpError("mismatch in region argument types ")
           << arg0Type << " and " << arg1Type;
  }
  auto yieldOp = cast<linalg_ext::YieldOp>(body->getTerminator());
  if (yieldOp->getNumOperands() != 1) {
    return yieldOp.emitOpError("expected region to yield a single value");
  }
  auto yieldedType = yieldOp->getOperand(0).getType();
  if (yieldedType != arg0Type) {
    return yieldOp.emitOpError("mismatch in type of yielded value ")
           << yieldedType << " and argument of the region " << arg0Type;
  }
  return success();
}

//===----------------------------------------------------------------------===//
// SortOp
//===----------------------------------------------------------------------===//

void SortOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  SmallVector<Value> inputBuffers = getInputBufferOperands();
  SmallVector<Value> outputBuffers = getOutputBufferOperands();
  getEffectsImpl(effects, getOperation()->getResults(), inputBuffers,
                 outputBuffers);
}

static LogicalResult verifySortOp(SortOp op) {
  if (op.getNumInputs()) {
    return op.emitOpError("does not expect to take any inputs");
  }

  Block &block = op.region().front();
  size_t numOutputs = op.getNumOutputs();
  if (block.getNumArguments() != 2 * numOutputs) {
    return op.emitOpError("region block should have ")
           << 2 * numOutputs << " arguments";
  }

  int rank = op.getRank(op.getOutputOperand(0));
  if (rank > 1 && !op.dimensionAttr()) {
    return op.emitOpError("dimension must be specified if rank > 1");
  }
  int dimension = 0;
  if (op.dimensionAttr()) {
    dimension = op.dimension().getValue();
  }
  if (dimension < 0 || dimension >= rank) {
    return op.emitOpError("dimension must be within (0, ") << rank << "]";
  }

  for (auto indexedOperand : llvm::enumerate(op.inputs())) {
    int index = indexedOperand.index();
    Type elemType =
        indexedOperand.value().getType().cast<ShapedType>().getElementType();
    for (int i : {2 * index, 2 * index + 1}) {
      Type argType = block.getArgument(i).getType();
      if (argType != elemType) {
        return op.emitOpError("region block argument #")
               << i << " should be of type " << elemType << " but got "
               << argType;
      }
    }
  }

  auto yieldOp = cast<YieldOp>(block.getTerminator());
  if (yieldOp.getNumOperands() != 1) {
    return op.emitOpError("should yield exactly one operand");
  }
  auto ty = yieldOp.getOperand(0).getType().dyn_cast<IntegerType>();
  if (!ty || ty.getWidth() != 1) {
    return op.emitOpError("should yield i1 type");
  }

  return success();
}

}  // namespace linalg_ext
}  // namespace iree_compiler
}  // namespace mlir

#define GET_OP_CLASSES
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.cpp.inc"
