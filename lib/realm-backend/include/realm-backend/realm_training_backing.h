#ifndef _FLEXFLOW_REALM_BACKEND_REALM_TRAINING_BACKING_H
#define _FLEXFLOW_REALM_BACKEND_REALM_TRAINING_BACKING_H

#include "realm-backend/realm_tensor_backing.h"
#include "realm-backend/realm_args_backing.h"
#include "local-execution/task_registry.h"
#include "op-attrs/ops/loss_functions/loss_attrs.dtg.h"
#include "pcg/computation_graph.dtg.h"
#include "pcg/optimizer_attrs.dtg.h"
#include "local-execution/optimizer_tensor_source.h"

namespace FlexFlow {

using PerLayerElapsedTime =
    std::unordered_map<layer_guid_t, std::optional<float>>;

struct LocalTrainingBacking {
  LocalTrainingBacking(Allocator const &,
                       ComputationGraph const &,
                       RuntimeArgConfig const &);
  void register_and_allocate_layer(layer_guid_t const &);
  void allocate_layer_optimizer_tensors(layer_guid_t const &,
                                        OptimizerAttrs const &);

  void execute_init(layer_guid_t const &);
  std::optional<float> execute_forward(layer_guid_t const &);
  void compute_loss(LossAttrs const &loss_attrs,
                    tensor_guid_t const &logit_tensor,
                    loss_tensor_t const &label_tensor);
  std::optional<float> execute_backward(layer_guid_t const &);
  void execute_update(layer_guid_t const &, OptimizerAttrs const &);

  TaskArgumentAccessor
      get_task_arg_accessor(TaskInvocation const &) const;

  TaskInvocation lower_to_task_invocation(OpTaskInvocation const &, layer_guid_t const &) const;

  LocalTensorBacking local_tensor_backing;
  LocalArgsBacking local_args_backing;

private:
  DeviceSpecificDeviceStates call_init_task_impl(task_id_t,
                                                 TaskArgumentAccessor const &);
  std::optional<float> call_task_impl(task_id_t, TaskArgumentAccessor);

private:
  Allocator allocator;
  ComputationGraph computation_graph;
  TaskRegistry task_registry;

  // optimizer
  OptimizerTensorSource optimizer_tensor_source;
  std::unordered_map<layer_guid_t, std::vector<optimizer_tensor_t>> layer_optimizer_tensor_ids;
};

} // namespace FlexFlow

#endif
