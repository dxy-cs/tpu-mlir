//===----------------------------------------------------------------------===//
//
// Copyright (c) 2020-2030 by Sophgo Technologies Inc. All rights reserved.
//
// Licensed under the Apache License v2.0.
// See http://www.apache.org/licenses/LICENSE-2.0 for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Backend/BM168x/BM1684x.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Support/Helper/Module.h"
#include "tpu_mlir/Support/Helper/Quant.h"
#include "tpu_mlir/Support/MathUtils.h"

using namespace mlir;
using namespace tpu_mlir;
using namespace tpu_mlir::helper;
using namespace tpu_mlir::backend;

// =========================================
// GlobalGenInterface
// =========================================

void tpu::DequantAxisOp::codegen_global_int8_bm1684x() {
  dequant_int_param_t param = {0};
  int64_t n, c, h, w;
  Module::getNCHW(input(), n, c, h, w);
  param.input_addr = Module::getAddress(input());
  param.output_addr = Module::getAddress(output());
  param.n = (int)n;
  param.c = (int)c;
  param.h = (int)h;
  param.w = (int)w;

  param.dequant_addr = Module::getAddress(quant());
  param.is_perchannel = true;
  param.lshift = quant_mode() == 0 ? 0 : lshift().getValue();
  param.mode = quant_mode();
  param.input_dtype = BM168x::getDataType(input());
  param.output_dtype = BM168x::getDataType(output());
  BM1684x::instance().call_global_func("backend_api_dequant_int_global", &param,
                                       sizeof(param));
}

void tpu::DequantAxisOp::codegen_global_float_bm1684x() {
  codegen_global_int8_bm1684x();
}

// =========================================
// LocalGenInterface
// =========================================

int64_t tpu::DequantAxisOp::getBufferSize_bm1684x(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_hslice, int64_t out_nslice, int64_t out_hslice) {
  if (quant_mode() == 1) {
    return in_lmem_bytes;
  }
}

void tpu::DequantAxisOp::codegen_local_int8_bm1684x(int64_t n_step,
                                                    int64_t h_step) {
  dequant_int_param_t param = {0};
  int64_t n, c, h, w;
  Module::getNCHW(input(), n, c, h, w);
  auto in_gi = LocalGenInterface::getGroupInfo(input(), n_step, h_step);
  auto gi = getGroupInfo(n_step, h_step);
  param.input_addr = (uint32_t)in_gi.out_addr;
  param.output_addr = (uint32_t)gi.out_addr;
  param.buffer_local_addr = (uint32_t)gi.buffer_addr;
  param.n = gi.n_slice;
  param.c = c;
  param.h = gi.h_slice;
  param.w = w;

  auto dequant_gi = LocalGenInterface::getGroupInfo(quant(), n_step, h_step);
  param.dequant_addr = (uint32_t)dequant_gi.out_addr;
  param.is_perchannel = true;

  param.input_dtype = BM168x::getDataType(input());
  param.output_dtype = BM168x::getDataType(output());
  param.lshift = quant_mode() == 0 ? 0 : lshift().getValue();
  param.mode = quant_mode();
  BM1684x::instance().call_local_func("backend_api_dequant_int_local", &param,
                                      sizeof(param));
}

void tpu::DequantAxisOp::codegen_local_float_bm1684x(int64_t n_step,
                                                     int64_t h_step) {
  llvm_unreachable("support later");
}
