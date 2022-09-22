//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "../Lowering.h"

using namespace tpu_mlir;
using namespace tpu_mlir::helper;
using namespace mlir;

void top::PermuteOp::lowering_int8_bm1684x(PatternRewriter &rewriter,
                                           bool asymmetric) {
  lowering_common_int8<tpu::PermuteOp>(rewriter, getOperation(), asymmetric);
}

void top::PermuteOp::lowering_f32_bm1684x(PatternRewriter &rewriter) {
  lowering_common_float<tpu::PermuteOp>(rewriter, getOperation());
}

void top::PermuteOp::lowering_bf16_bm1684x(PatternRewriter &rewriter) {
  lowering_common_float<tpu::PermuteOp, BFloat16Type>(rewriter, getOperation());
}

void top::PermuteOp::lowering_f16_bm1684x(PatternRewriter &rewriter) {
  lowering_common_float<tpu::PermuteOp, Float16Type>(rewriter, getOperation());
}

void top::PermuteOp::lowering_quant_bm1684x(PatternRewriter &rewriter) {
  lowering_common<tpu::PermuteOp>(rewriter, getOperation(), output().getType());
}
