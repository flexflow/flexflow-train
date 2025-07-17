#ifndef _FLEXFLOW_LOCAL_EXECUTION_MODEL_TRAINING_INSTANCE_H
#define _FLEXFLOW_LOCAL_EXECUTION_MODEL_TRAINING_INSTANCE_H

#include "local-execution/local_training_backing.h"
#include "op-attrs/ops/loss_functions/loss_attrs.dtg.h"
#include "pcg/tensor_guid_t.dtg.h"
#include "task-spec/loss_tensor_guid_t.dtg.h"

namespace FlexFlow {

struct ModelTrainingInstance {
  ModelTrainingInstance(Allocator const &,
                        LocalTrainingBacking const &,
                        LossAttrs const &,
                        OptimizerAttrs const &);

  Allocator allocator;
  LocalTrainingBacking training_backing;
  LossAttrs loss_attrs;
  OptimizerAttrs optimizer_attrs;

public:
  std::unordered_map<layer_guid_t, std::optional<milliseconds_t>> forward();
  std::unordered_map<layer_guid_t, std::optional<milliseconds_t>> backward();
  void update();
  GenericTensorAccessorR get_loss_tensor_accessor() const;
};

} // namespace FlexFlow

#endif
