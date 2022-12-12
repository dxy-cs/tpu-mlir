#pragma once

#include "tpu_mlir/Dialect/Tpu/Transforms/LayerGroup/BasicTimeStep.h"

namespace tpu_mlir {
namespace tpu {

struct LgOptions {
  bool dyn_compile;
  int64_t opt;
};

struct LgPassIR {
  LgPassIR(){};
  ~LgPassIR() { clear(); };

  /**
   * Clear information of layer group IR
   */
  void clear();

  /**
   * @brief the layers in the current subnet graph
   */
  std::vector<Operation *> subnet_ops;

  /**
   * @brief the layers in the current subnet graph
   */
  std::vector<Value> subnet_values;

  /**
   * @brief the layer groups.
   * lg_infos.size() means the number of groups.
   * lg_infos[i].ids.size() means the number of layers in the i-th group
   */
  std::vector<LgInfo> lg_infos;

  /**
   * @brief time step of layer groups
   * time_steps.size() == lg_infos.size()
   * time_steps[i] means the time step of the i-th group
   */
  std::vector<BasicTimeStepPtr> time_steps;

  /**
   * @brief shape split sections of layer groups
   * shape_secs.size() == lg_infos.size()
   * shape_secs[i] means the shape split sections of the i-th group
   */
  std::vector<shape_secs_t> shape_secs;
};

class LgPass {
public:
  LgPass() {}
  virtual ~LgPass() {}

  virtual bool run(LgPassIR *pass_ir) = 0;
  virtual std::string name() = 0;
  virtual std::string brief() { return ""; }
};

/// Pass manager of layer group optimization
class LgPassManager {
public:
  LgPassManager() {}
  ~LgPassManager() {}

  void add_pass(std::unique_ptr<LgPass> pass);
  void run(LgPassIR *pass_ir);

private:
  std::vector<std::unique_ptr<LgPass>> passes;
};

/// Layer group optimizer
class LgOptimizer {
public:
  LgOptimizer() {}
  virtual ~LgOptimizer() {}

  virtual void manage_passes(std::shared_ptr<LgPassManager> pm,
                             const LgOptions &options) = 0;
  virtual std::string brief() = 0;
};

using LgOptimizerMap = std::map<std::string, LgOptimizer *>;

const LgOptimizerMap &get_registered_optimizers();

struct LgOptimizerReg {
  LgOptimizerReg(const std::string &name,
                 std::shared_ptr<LgOptimizer> optimizer);
};

#define REGISTER_LG_OPTIMIZER(name, optimizer)                                 \
  static std::shared_ptr<LgOptimizer> name##_opt_inst(new optimizer());        \
  static LgOptimizerReg name##_lg_reg(#name, name##_opt_inst)

} // namespace tpu
} // namespace tpu_mlir
