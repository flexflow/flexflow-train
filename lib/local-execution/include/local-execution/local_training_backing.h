#ifndef _FLEXFLOW_LOCAL_EXECUTION_LOCAL_TRAINING_BACKING_H
#define _FLEXFLOW_LOCAL_EXECUTION_LOCAL_TRAINING_BACKING_H

#include "local-execution/allocated_tensors.dtg.h"
#include "local-execution/local_args_backing.h"
#include "local-execution/local_tensor_backing.h"
#include "local-execution/optimizer_tensor_source.h"
#include "local-execution/task_registry.h"
#include "op-attrs/ops/loss_functions/loss_attrs.dtg.h"
#include "pcg/computation_graph.dtg.h"
#include "pcg/optimizer_attrs.dtg.h"

namespace FlexFlow {

struct LocalTrainingBacking {
  LocalTrainingBacking(Allocator &,
                       AllocatedTensors const &,
                       GradientTensorSource &,
                       ComputationGraph const &,
                       RuntimeArgConfig const &);

  LocalTrainingBacking(Allocator &,
                       AllocatedTensors const &,
                       GradientTensorSource &,
                       OptimizerTensorSource &,
                       ComputationGraph const &,
                       RuntimeArgConfig const &,
                       OptimizerAttrs const &);

public:
  ComputationGraph computation_graph;
  TaskRegistry task_registry;
  LocalTensorBacking local_tensor_backing;
  LocalArgsBacking local_args_backing;
};

LocalArgsBacking initialize_args_backing(TaskRegistry const &,
                                         ComputationGraph const &,
                                         RuntimeArgConfig const &,
                                         LocalTensorBacking const &,
                                         Allocator &);

std::optional<float> call_task_impl(TaskRegistry const &,
                                    task_id_t const &task_id,
                                    TaskArgumentAccessor const &acc);

std::optional<float> execute_forward(LocalTrainingBacking const &,
                                     layer_guid_t const &,
                                     Allocator &);
std::optional<float> execute_backward(LocalTrainingBacking const &,
                                      layer_guid_t const &,
                                      Allocator &);
void compute_loss(LocalTrainingBacking const &,
                  LossAttrs const &,
                  tensor_guid_t const &logit_tensor,
                  loss_tensor_t const &label_tensor,
                  Allocator &);
void execute_update(LocalTrainingBacking const &,
                    layer_guid_t const &,
                    OptimizerAttrs const &,
                    Allocator &);

TaskArgumentAccessor get_task_arg_accessor(LocalTensorBacking const &,
                                           LocalArgsBacking const &,
                                           TaskInvocation const &,
                                           Allocator &);

} // namespace FlexFlow

#endif
