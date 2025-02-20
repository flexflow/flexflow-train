#ifndef _FLEXFLOW_LOCAL_EXECUTION_MODEL_TRAINING_INSTANCE_H
#define _FLEXFLOW_LOCAL_EXECUTION_MODEL_TRAINING_INSTANCE_H

#include "local-execution/local_training_backing.h"
#include "op-attrs/ops/loss_functions/loss_attrs.dtg.h"
#include "pcg/tensor_guid_t.dtg.h"
#include "task-spec/loss_tensor_t.dtg.h"

namespace FlexFlow {

using PerLayerElapsedTime =
    std::unordered_map<layer_guid_t, std::optional<float>>;

struct ModelTrainingInstance {
  ModelTrainingInstance(LocalTrainingBacking const &,
                        tensor_guid_t const &logit_tensor,
                        loss_tensor_t const &label_tensor,
                        LossAttrs const &,
                        OptimizerAttrs const &);

  LocalTrainingBacking training_backing;
  tensor_guid_t logit_tensor;
  loss_tensor_t label_tensor;
  LossAttrs loss_attrs;
  OptimizerAttrs optimizer_attrs;
};

PerLayerElapsedTime forward(ModelTrainingInstance const &);
PerLayerElapsedTime backward(ModelTrainingInstance const &);
void update(ModelTrainingInstance &);

} // namespace FlexFlow

#endif
