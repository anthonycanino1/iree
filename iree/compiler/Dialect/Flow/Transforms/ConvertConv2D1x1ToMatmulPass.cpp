// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

namespace {

// Converts linalg.conv_2d_input_nhwc_filter_nhwc op to linalg.matmul
class Convert1x1ConvolutionMatmulOp
    : public OpRewritePattern<linalg::ConvInputNHWCFilterHWCFOp> {
 public:
  using OpRewritePattern<linalg::ConvInputNHWCFilterHWCFOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::ConvInputNHWCFilterHWCFOp convOp,
                                PatternRewriter &rewriter) const override {
    ShapedType inputShapeType =
        convOp.getInputOperand(0)->get().getType().cast<ShapedType>();
    ShapedType filterShapeType =
        convOp.getInputOperand(1)->get().getType().cast<ShapedType>();
    ShapedType outputShapeType =
        convOp.getOutputOperand(0)->get().getType().cast<ShapedType>();

    auto inputShape = inputShapeType.getShape();
    auto filterShape = filterShapeType.getShape();
    auto outputShape = outputShapeType.getShape();

    if (filterShape[0] != 1 || filterShape[1] != 1) return failure();

    // TODO(ataei): Support conversion to linalg.batch_matmul.
    if (inputShape[0] != 1) return failure();

    if (!llvm::all_of(convOp.strides(), [](APInt element) {
          return element.getSExtValue() == 1;
        }))
      return failure();
    if (!llvm::all_of(convOp.dilations(), [](APInt element) {
          return element.getSExtValue() == 1;
        }))
      return failure();

    SmallVector<linalg::ReassociationIndices, 4> reassociationIndices = {
        {0, 1, 2}, {3}};

    auto reshapedInputType =
        RankedTensorType::get({inputShape[1] * inputShape[2], inputShape[3]},
                              inputShapeType.getElementType());

    auto reshapedFilterType = RankedTensorType::get(
        {filterShape[2], filterShape[3]}, filterShapeType.getElementType());

    auto reshapedOutputType =
        RankedTensorType::get({outputShape[1] * outputShape[2], outputShape[3]},
                              outputShapeType.getElementType());

    Value input = convOp.getInputOperand(0)->get();
    Value filter = convOp.getInputOperand(1)->get();
    Value output = convOp.getOutputOperand(0)->get();
    auto loc = convOp.getLoc();

    Value reshapedInput = rewriter.create<linalg::TensorCollapseShapeOp>(
        loc, reshapedInputType, input, reassociationIndices);
    Value reshapedFilter = rewriter.create<linalg::TensorCollapseShapeOp>(
        loc, reshapedFilterType, filter, reassociationIndices);
    Value reshapedOutput = rewriter.create<linalg::TensorCollapseShapeOp>(
        loc, reshapedOutputType, output, reassociationIndices);

    auto matmulResult = rewriter.create<linalg::MatmulOp>(
        loc, reshapedOutputType, ArrayRef<Value>{reshapedInput, reshapedFilter},
        ArrayRef<Value>{reshapedOutput});

    auto reshapedResult = rewriter.create<linalg::TensorExpandShapeOp>(
        loc, outputShapeType, matmulResult.getResults()[0],
        reassociationIndices);

    rewriter.replaceOp(convOp, ArrayRef<Value>{reshapedResult});

    return success();
  }
};

struct ConvertConv2D1x1ConvToMatmulPass
    : public ConvertConv2D1x1ConvToMatmulBase<
          ConvertConv2D1x1ConvToMatmulPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    OwningRewritePatternList patterns(&getContext());
    patterns.insert<Convert1x1ConvolutionMatmulOp>(context);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};
}  // namespace

std::unique_ptr<OperationPass<FuncOp>> createConvertConv2D1x1ToMatmulPass() {
  return std::make_unique<ConvertConv2D1x1ConvToMatmulPass>();
}

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
