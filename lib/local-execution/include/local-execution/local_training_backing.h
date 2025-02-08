#ifndef _FLEXFLOW_LOCAL_EXECUTION_LOCAL_TRAINING_BACKING_H
#define _FLEXFLOW_LOCAL_EXECUTION_LOCAL_TRAINING_BACKING_H

#include "local-execution/local_args_backing.h"
#include "local-execution/local_tensor_backing.h"
#include "local-execution/optimizer_tensor_source.h"
#include "local-execution/task_registry.h"
#include "op-attrs/ops/loss_functions/loss_attrs.dtg.h"
#include "pcg/computation_graph.dtg.h"
#include "pcg/optimizer_attrs.dtg.h"

namespace FlexFlow {

struct LocalTrainingBacking {
  LocalTrainingBacking(Allocator const &,
                       ComputationGraph const &,
                       LocalTensorBacking const &,
                       LocalArgsBacking const &);

public:
  LocalTensorBacking local_tensor_backing;
  LocalArgsBacking local_args_backing;

  Allocator allocator;
  ComputationGraph computation_graph;
  TaskRegistry task_registry;

  GradientTensorSource gradient_tensor_source;
};

DeviceSpecificDeviceStates call_init_task_impl(TaskRegistry const &,
                                               task_id_t task_id,
                                               TaskArgumentAccessor const &acc);

std::optional<float> call_task_impl(TaskRegistry const &,
                                    task_id_t task_id,
                                    TaskArgumentAccessor acc);

void execute_init(LocalTrainingBacking &, layer_guid_t const &);
std::optional<float> execute_forward(LocalTrainingBacking &,
                                     layer_guid_t const &);
std::optional<float> execute_backward(LocalTrainingBacking &,
                                      layer_guid_t const &);
void compute_loss(LocalTrainingBacking &,
                  LossAttrs const &,
                  tensor_guid_t const &logit_tensor,
                  loss_tensor_t const &label_tensor);
void execute_update(LocalTrainingBacking &,
                    layer_guid_t const &,
                    OptimizerAttrs const &);

TaskArgumentAccessor get_task_arg_accessor(LocalTensorBacking const &,
                                           LocalArgsBacking const &,
                                           TaskInvocation const &,
                                           Allocator &);

} // namespace FlexFlow

#endif
