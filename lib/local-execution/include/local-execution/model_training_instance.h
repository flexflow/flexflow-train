#ifndef _FLEXFLOW_LOCAL_EXECUTION_MODEL_TRAINING_INSTANCE_H
#define _FLEXFLOW_LOCAL_EXECUTION_MODEL_TRAINING_INSTANCE_H

#include "local-execution/local_atomic_tensor_backing.dtg.h"
#include "local-execution/local_training_backing.h"
#include "op-attrs/ops/loss_functions/loss_attrs.dtg.h"
#include "pcg/tensor_guid_t.dtg.h"
#include "task-spec/training_symbolic_computation_graph.dtg.h"

namespace FlexFlow {

struct ModelTrainingInstance {
  ModelTrainingInstance(Allocator const &,
                        LossAttrs const &,
                        OptimizerAttrs const &);

public:
  std::unordered_map<layer_guid_t, std::optional<milliseconds_t>> forward();
  std::unordered_map<layer_guid_t, std::optional<milliseconds_t>> backward();
  void update();
  GenericTensorAccessorR get_loss_tensor_accessor() const;

private:
  Allocator allocator;
  LossAttrs loss_attrs;
  OptimizerAttrs optimizer_attrs;
  TrainingSymbolicComputationGraphFromCgConversion symbolic_cg;
  LocalTensorBacking local_tensor_backing;
  LocalAtomicTensorBacking local_atomic_tensor_backing;
  LocalTaskRegistry local_task_registry;
  RuntimeArgConfig runtime_arg_config;
};

} // namespace FlexFlow

#endif
