#include "sophgo/Dialect/Top/IR/TopOps.h"
#include "sophgo/Dialect/Tpu/IR/TpuOps.h"
#include "sophgo/Support/MathUtils.h"
#include "sophgo/Support/Helper/Quant.h"

using namespace mlir;
using namespace sophgo;
using namespace sophgo::helper;

Value top::AddOp::quantize_int8_bm1686() {
  auto op = getOperation();
  OpBuilder builder(op);
  std::vector<Value> operands;
  const int nInputs = op->getNumOperands();
  std::vector<int64_t> rshift_v(nInputs);
  std::vector<int64_t> multiplier_v(nInputs, 1);
  std::shared_ptr<std::vector<double>> coeff_v;
  int64_t o_zp;
  double o_scale;
  Quant::getScaleAndZeroPoint(output(), o_scale, o_zp);

  if (coeff().hasValue()) {
    coeff_v = Module::getF64Array(coeff().getValue());
  } else {
    coeff_v = std::make_shared<std::vector<double>>(nInputs, 1.0);
  }

  double bias = 0;
  int max_shifti = -32;
  double scale;
  int64_t zeropoint;
  for (int i = 0; i < nInputs; i++) {
    auto input = op->getOperand(i);
    operands.push_back(input);
    Quant::getScaleAndZeroPoint(input, scale, zeropoint);
    int scalei, shifti;
    auto scale_f = scale / o_scale;
    bias += scale_f * zeropoint;
    get_scale_and_shift(coeff_v->at(i) * scale_f, scalei, shifti, 8);
    multiplier_v[i] = scalei;
    rshift_v[i] = shifti;
  }

  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder.getNamedAttr("name", nameAttr()));
  attrs.push_back(builder.getNamedAttr("do_relu", do_reluAttr()));
  attrs.push_back(builder.getNamedAttr("multipliers",
                                       builder.getI64ArrayAttr(multiplier_v)));
  attrs.push_back(
      builder.getNamedAttr("rshifts", builder.getI64ArrayAttr(rshift_v)));
  attrs.push_back(
      builder.getNamedAttr("rectified_bias", builder.getF64FloatAttr(bias)));
  auto newOp = builder.create<tpu::AddOp>(op->getLoc(), output().getType(),
                                          ArrayRef<Value>{operands},
                                          ArrayRef<NamedAttribute>{attrs});
  Quant::setQuantInt8Type(newOp.output(), true);
  return newOp.output();
}