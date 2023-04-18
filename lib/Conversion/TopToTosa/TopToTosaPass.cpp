//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include "tpu_mlir/Conversion/TopToTosa/OpLowering.h"
#include "tpu_mlir/Conversion/Conversion.h"
#include "tpu_mlir/Support/Module.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTTOPTOTOSA
#include "tpu_mlir/Conversion/Passes.h.inc"
} // namespace mlir

namespace tpu_mlir {

struct LowerTopWeightOp : public OpRewritePattern<top::WeightOp> {
public:
  LowerTopWeightOp(MLIRContext *ctx, bool include_weight)
      : OpRewritePattern(ctx), include_weight(include_weight) {}

  LogicalResult matchAndRewrite(top::WeightOp op,
                                PatternRewriter &rewriter) const override {
    assert(op->getNumResults() == 1);
    auto outType = change_dataformat(op->getResult(0).getType());
    auto has_weight = include_weight;
    for (auto user : op.getOutput().getUsers()) {
      if (isa<tosa::TransposeOp>(user)) {
        has_weight = true;
      }
    }
    if (has_weight) {
      auto valptr = op.read_as_float();
      auto new_val = change_weight(valptr, op->getResult(0).getType());
      auto attr = DenseElementsAttr::get(
          outType, llvm::makeArrayRef(new_val, valptr->size()));
      rewriter.replaceOpWithNewOp<mlir::tosa::ConstOp>(op, outType, attr);
    } else {
      // auto out_shape = outType.cast<RankedTensorType>().getShape();
      // auto out_ty = RankedTensorType::get(out_shape, rewriter.getF32Type());
      // attr = DenseElementsAttr::get(out_ty, llvm::ArrayRef<float>());
      auto attr = DenseElementsAttr::get(
          RankedTensorType::get({}, rewriter.getI64Type()),
          llvm::ArrayRef<int64_t>({0}));
      rewriter.replaceOpWithNewOp<mlir::tosa::ConstOp>(op, outType, attr);
    }
    return success();
  }

private:
  bool include_weight;
};

struct ConvertTopToTosa
    : public ::impl::ConvertTopToTosaBase<ConvertTopToTosa> {
public:
  void runOnOperation() override {
    module_ = getOperation();
    ctx_ = &getContext();
    mainFunc_ = module::getMainFuncOp();

    RewritePatternSet patterns(ctx_);
    ConversionTarget target(*ctx_);
    target.addLegalDialect<mlir::tosa::TosaDialect, mlir::func::FuncDialect>();

    // Lower TOP Ops
    patterns.add<LowerTopWeightOp>(patterns.getContext(), includeWeight);
    populateTopToTosaConversionPatterns(&patterns);
    auto config = GreedyRewriteConfig();
    config.maxIterations = 0;
    applyPatternsAndFoldGreedily(module_, std::move(patterns), config);

    module::updateModuleTypes();
    module::setState(module::State::TOSA_F32);
  }

protected:
  ModuleOp module_;
  FuncOp mainFunc_;
  MLIRContext *ctx_;
};

std::unique_ptr<Pass> createConvertTopToTosa() {
  return std::make_unique<ConvertTopToTosa>();
}

} // namespace tpu_mlir
