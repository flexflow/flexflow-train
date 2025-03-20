#ifndef _FLEXFLOW_REALM_BACKEND_REALM_TRAINING_BACKING_H
#define _FLEXFLOW_REALM_BACKEND_REALM_TRAINING_BACKING_H

#include "local-execution/optimizer_tensor_source.h"
#include "local-execution/task_registry.h"
#include "op-attrs/ops/loss_functions/loss_attrs.dtg.h"
#include "pcg/computation_graph.dtg.h"
#include "pcg/optimizer_attrs.dtg.h"
#include "local-execution/allocated_tensors.h"
#include "local-execution/unallocated_tensors.h"
#include "realm-backend/driver.h"
#include "realm-backend/realm_allocator.h"
#include "realm-backend/realm_args_backing.h"
#include "realm-backend/realm_tensor_backing.h"
#include "realm-backend/task_wrapper.h"

namespace FlexFlow {

using PerLayerElapsedTime =
    std::unordered_map<layer_guid_t, std::optional<float>>;

struct RealmTrainingBacking {
  RealmTrainingBacking(Realm::Processor, 
    std::vector<Realm::Processor> const &, 
    std::vector<Allocator> const &,
                      AllocatedTensors const &,
                      GradientTensorSource &,
                       ComputationGraph const &, RuntimeArgConfig const &);

  RealmTrainingBacking(Realm::Processor, 
    std::vector<Realm::Processor> const &, 
    std::vector<Allocator> const &,
    AllocatedTensors const &,
    GradientTensorSource &,
    OptimizerTensorSource &,
                       ComputationGraph const &, RuntimeArgConfig const &,
                       OptimizerAttrs const &);

public:
  // runtime
  Realm::Processor master_proc;
  Realm::Event master_event;
  Realm::Memory master_mem;
  std::vector<Realm::Processor> worker_procs;
  std::vector<Realm::Event> worker_events;
  std::vector<Allocator> allocators;

  ComputationGraph computation_graph;
  TaskRegistry task_registry;

  RealmTensorBacking realm_tensor_backing;
  RealmArgsBacking realm_args_backing;
};

TaskRegistry construct_task_registry_and_register_tasks_for_realm(
    ComputationGraph const &, std::vector<Realm::Processor> const &);

RealmArgsBacking initialize_args_backing(RealmTrainingBacking *,
                                        ComputationGraph const &,
                                        RuntimeArgConfig const &);

void execute_init(RealmTrainingBacking &, layer_guid_t const &);
Future<float> execute_forward(RealmTrainingBacking &,
                              layer_guid_t const &);
Future<float> execute_backward(RealmTrainingBacking &,
                              layer_guid_t const &);
Future<void> compute_loss(RealmTrainingBacking &, LossAttrs const &,
                          tensor_guid_t const &logit_tensor,
                          loss_tensor_t const &label_tensor);
Future<void> execute_update(RealmTrainingBacking &, layer_guid_t const &,
                            OptimizerAttrs const &);

TaskArgumentAccessor get_task_arg_accessor(RealmTensorBacking const &,
                                           RealmArgsBacking const &,
                                           TaskInvocation const &,
                                           Allocator &);

} // namespace FlexFlow

#endif
