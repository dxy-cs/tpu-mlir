//===----------------------------------------------------------------------===//
//
// Copyright (c) 2020-2030 by Sophgo Technologies Inc. All rights reserved.
//
// Licensed under the Apache License v2.0.
// See http://www.apache.org/licenses/LICENSE-2.0 for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Support/Dnnl/Dnnl.h"
#include "tpu_mlir/Support/Helper/Quant.h"
#include "tpu_mlir/Support/Helper/Module.h"
#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Support/Float16.h"

using namespace tpu_mlir;
using namespace tpu_mlir::helper;
using namespace mlir;

void tpu::AvgPoolOp::parseParam(int64_t &n, int64_t &c, int64_t &ih,
                                int64_t &iw, int64_t &oh, int64_t &ow,
                                int64_t &kh, int64_t &kw, int64_t &sh,
                                int64_t &sw, int64_t &pt, int64_t &pb,
                                int64_t &pl, int64_t &pr, int64_t &pad_value,
                                bool &relu, bool &is_global,
                                bool &count_include_pad) {
  Module::getNCHW(input(), n, c, ih, iw);
  int64_t on, oc;
  Module::getNCHW(output(), on, oc, oh, ow);
  assert(on == n && oc == c);
  auto kernel = Module::getI64Array(kernel_shape());
  kh = kernel->at(0);
  kw = kernel->at(1);
  auto stride = Module::getI64Array(strides());
  sh = stride->at(0);
  sw = stride->at(1);
  relu = do_relu();
  auto pad = Module::getI64Array(pads());
  pt = pad->at(0);
  pl = pad->at(1);
  pb = pad->at(2);
  pr = pad->at(3);
  is_global = false;
  if (kh == ih && kw == iw && oh == 1 && ow == 1) {
    is_global = true;
  }
  pad_value = this->pad_value();
  count_include_pad = this->count_include_pad();
  if (pt == 0 && pb == 0 && pl == 0 && pr == 0) {
    // no pad
    count_include_pad = true;
  }
}

LogicalResult tpu::AvgPoolOp::init(InferenceParameter &p) {
  auto pooling = new Pooling();
  int64_t n, c, ih, iw, oh, ow, kh, kw, sh, sw, pt, pb, pl, pr, pad_value;
  bool relu, is_global, count_include_pad;
  parseParam(n, c, ih, iw, oh, ow, kh, kw, sh, sw, pt, pb, pl, pr, pad_value,
             relu, is_global, count_include_pad);

  int izp = 0;
  auto dtype = input().getType().cast<RankedTensorType>().getElementType();
  if (dtype.isa<quant::UniformQuantizedType>()) {
    izp = dtype.cast<quant::UniformQuantizedType>().getZeroPoint();
  }

  pooling->setup(p.inputs[0], p.outputs[0], n, c, ih, iw, oh, ow, kh, kw, sh,
                 sw, pt, pb, pl, pr, true, count_include_pad, izp, pad_value);
  p.handle = (void *)pooling;
  return success();
}

void tpu::AvgPoolOp::deinit(InferenceParameter &p) {
  if (p.handle != nullptr) {
    auto pooling = (Pooling *)p.handle;
    delete pooling;
    p.handle = nullptr;
  }
  return;
}

LogicalResult tpu::AvgPoolOp::inference(InferenceParameter &p) {
  if (p.handle == nullptr) {
    return failure();
  }
  auto pooling = (Pooling *)p.handle;
  pooling->run();
  auto out_type = Module::getStorageType(output());
  auto num_elem = Module::getNumElements(output());
  if (out_type.isInteger(8)) {
    auto i_qtype = Quant::getUniformQuantizedType(input());
    auto o_qtype = Quant::getUniformQuantizedType(output());
    auto multi = multiplier();
    auto rs = rshift();
#pragma omp parallel for schedule(static, omp_schedule(num_elem))
    for (int64_t i = 0; i < num_elem; ++i) {
      p.outputs[0][i] = applyMultiplierAndRShift(
          std::round(p.outputs[0][i] * pooling->kh * pooling->kw), multi, rs);
      p.outputs[0][i] = out_type.isUnsignedInteger(8)
                            ? Quant::to_uint8(p.outputs[0][i])
                            : Quant::to_int8(p.outputs[0][i]);
    }
  } else if (out_type.isa<FloatType>()) {
    if (do_relu()) {
      function_relu(p.outputs[0], p.outputs[0], num_elem);
    }
    if (out_type.isBF16()) {
      f32_to_bf16(p.outputs[0], p.outputs[0], num_elem);
    } else if (out_type.isF16()) {
      f32_to_f16(p.outputs[0], p.outputs[0], num_elem);
    }
  }

  return success();
}

LogicalResult tpu::AvgPoolOp::LocalGenSupport() {
  int64_t n, c, ih, iw, oh, ow, kh, kw, sh, sw, pt, pb, pl, pr, pad_value;
  bool relu, is_global, count_include_pad;
  parseParam(n, c, ih, iw, oh, ow, kh, kw, sh, sw, pt, pb, pl, pr, pad_value,
             relu, is_global, count_include_pad);
  if (is_global == false && (sh > 15 || sw > 15)) {
    return failure();
  }
  return success();
}

LogicalResult tpu::AvgPoolOp::BackwardH(int64_t &in_idx, int64_t &in_slice,
                                        int64_t out_idx, int64_t out_slice) {
  int64_t n, c, ih, iw, oh, ow, kh, kw, sh, sw, pt, pb, pl, pr, pad_value;
  bool relu, is_global, count_include_pad;
  parseParam(n, c, ih, iw, oh, ow, kh, kw, sh, sw, pt, pb, pl, pr, pad_value,
             relu, is_global, count_include_pad);
  in_slice = (out_slice - 1) * sh + kh;
  in_idx = out_idx * sh - pt;
  LocalGenInterface::fixSlice(in_idx, in_slice, ih);
  return success();
}