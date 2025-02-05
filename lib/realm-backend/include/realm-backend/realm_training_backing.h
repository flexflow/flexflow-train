#ifndef _FLEXFLOW_REALM_BACKEND_REALM_TRAINING_BACKING_H
#define _FLEXFLOW_REALM_BACKEND_REALM_TRAINING_BACKING_H

#include "local-execution/optimizer_tensor_source.h"
#include "local-execution/task_registry.h"
#include "op-attrs/ops/loss_functions/loss_attrs.dtg.h"
#include "pcg/computation_graph.dtg.h"
#include "pcg/optimizer_attrs.dtg.h"
#include "realm-backend/driver.h"
#include "realm-backend/realm_allocator.h"
#include "realm-backend/realm_args_backing.h"
#include "realm-backend/realm_tensor_backing.h"
#include "realm-backend/task_wrapper.h"

namespace FlexFlow {

using PerLayerElapsedTime =
    std::unordered_map<layer_guid_t, std::optional<float>>;

struct RealmTrainingBacking {
  RealmTrainingBacking(ComputationGraph const &, RuntimeArgConfig const &,
                       Realm::Processor);
  void register_and_allocate_layer(layer_guid_t const &);
  void allocate_layer_optimizer_tensors(layer_guid_t const &,
                                        OptimizerAttrs const &);

  void execute_init(layer_guid_t const &);
  Future<std::optional<float>> execute_forward(layer_guid_t const &);
  Future<std::optional<float>> execute_backward(layer_guid_t const &);
  Future<void> execute_update(layer_guid_t const &, OptimizerAttrs const &);
  Future<void> compute_loss(LossAttrs const &loss_attrs,
                            tensor_guid_t const &logit_tensor,
                            loss_tensor_t const &label_tensor);

  TaskArgumentAccessor get_task_arg_accessor(TaskInvocation const &) const;

  TaskInvocation lower_to_task_invocation(OpTaskInvocation const &,
                                          layer_guid_t const &) const;

  ComputationGraph computation_graph;
  TaskRegistry task_registry;

  // runtime
  Realm::Processor master_proc;
  Realm::Memory master_mem;
  std::vector<Realm::Processor> worker_procs;
  std::unordered_map<Realm::Processor, Realm::Event> proc_events;
  std::vector<RealmAllocator> allocators;

  // storage
  RealmTensorBacking realm_tensor_backing;
  RealmArgsBacking realm_args_backing;
  OptimizerTensorSource optimizer_tensor_source;
  std::unordered_map<layer_guid_t, std::vector<optimizer_tensor_t>>
      layer_optimizer_tensor_ids;

private:
  std::optional<float> call_task_impl(task_id_t, TaskSignatureAndImpl,
                                      TaskArgumentAccessor);
};

} // namespace FlexFlow

#endif
