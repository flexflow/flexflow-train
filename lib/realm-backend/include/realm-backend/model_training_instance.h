#ifndef _FLEXFLOW_LOCAL_EXECUTION_MODEL_TRAINING_INSTANCE_H
#define _FLEXFLOW_LOCAL_EXECUTION_MODEL_TRAINING_INSTANCE_H

#include "realm-backend/realm_training_backing.h"
#include "op-attrs/ops/loss_functions/loss_attrs.dtg.h"
#include "pcg/tensor_guid_t.dtg.h"
#include "local-execution/loss_tensor_t.dtg.h"

namespace FlexFlow {

using PerLayerElapsedTime =
    std::unordered_map<layer_guid_t, std::optional<float>>;

struct ModelTrainingInstance {
  ModelTrainingInstance(ComputationGraph const &,
                        RuntimeArgConfig const &,
                        LossAttrs const &,
                        tensor_guid_t const &logit_tensor,
                        loss_tensor_t const &label_tensor,
                        OptimizerAttrs const &);

  void execute_init();
  PerLayerElapsedTime execute_forward();
  PerLayerElapsedTime execute_backward();
  void execute_update();

  ComputationGraph computation_graph;
  RealmTrainingBacking training_backing;
  LossAttrs loss_attrs;
  tensor_guid_t logit_tensor;
  loss_tensor_t label_tensor;
  OptimizerAttrs optimizer_attrs;
};

} // namespace FlexFlow

#endif
