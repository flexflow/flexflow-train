#ifndef _FLEXFLOW_LOCAL_EXECUTION_MODEL_TRAINING_INSTANCE_H
#define _FLEXFLOW_LOCAL_EXECUTION_MODEL_TRAINING_INSTANCE_H

#include "local-execution/local_training_backing.h"
#include "op-attrs/ops/loss_functions/loss_attrs.dtg.h"

namespace FlexFlow {

using PerLayerElapsedTime =
    std::unordered_map<layer_guid_t, std::optional<float>>;

struct ModelTrainingInstance {
  ModelTrainingInstance(Allocator const &,
                        ComputationGraph const &,
                        LayerTensorBackingMap const &allocated_forward_tensors,
                        TensorBackingMap const &allocated_non_graph_tensors,
                        RuntimeArgConfig const &,
                        LossAttrs const &,
                        reduced_tensor_t const &logit_tensor,
                        reduced_tensor_t const &label_tensor,
                        OptimizerAttrs const &);

  void execute_init();
  PerLayerElapsedTime execute_forward();
  PerLayerElapsedTime execute_backward();
  void execute_update();

  ComputationGraph computation_graph;
  LocalTrainingBacking training_backing;
  LossAttrs loss_attrs;
  reduced_tensor_t logit_tensor;
  reduced_tensor_t label_tensor;
  OptimizerAttrs optimizer_attrs;
};

} // namespace FlexFlow

#endif
