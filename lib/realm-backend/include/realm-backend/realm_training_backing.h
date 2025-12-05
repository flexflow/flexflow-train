#ifndef _FLEXFLOW_REALM_BACKEND_REALM_TRAINING_BACKING_H
#define _FLEXFLOW_REALM_BACKEND_REALM_TRAINING_BACKING_H

#include "local-execution/local_training_backing.dtg.h"
#include "op-attrs/ops/loss_functions/loss_attrs.dtg.h"
#include "pcg/optimizer_attrs.dtg.h"
#include "task-spec/training_computation_graph.dtg.h"
#include "task-spec/training_tensor_guid_t.dtg.h"
#include "utils/containers/generate_map.h"
#include "utils/units/milliseconds_t.h"
#include "realm-backend/driver.h"
#include "realm-backend/realm_allocator.h"
#include "realm-backend/task_wrapper.h"

namespace FlexFlow {

struct RealmRuntimeState {
  Realm::Processor master_proc;
  Realm::Event master_event;
  Realm::Memory master_mem;
  std::vector<Realm::Processor> worker_procs;
  std::vector<Realm::Event> worker_events;
  std::vector<Allocator> allocators;
};

LocalTrainingBacking make_realm_training_backing_for_computation_graph(
    RealmRuntimeState &runtime_state,
    std::unordered_map<training_tensor_guid_t, GenericTensorAccessorW> const
        &preallocated_tensors,
    TrainingComputationGraph const &training_computation_graph,
    RuntimeArgConfig const &runtime_arg_config,
    OptimizerAttrs const &optimizer_attrs);

void register_tasks_for_realm(LocalTaskRegistry const &, RealmRuntimeState &);

std::optional<DeviceSpecificDeviceStates>
    create_per_device_op_state(LocalTaskRegistry const &,
                               LocalTensorBacking const &,
                               RuntimeArgConfig const &,
                               RealmRuntimeState &,
                               TrainingLayerPlusContext const &);

Future<std::optional<milliseconds_t>> execute_forward(LocalTaskRegistry const &,
                                              LocalTensorBacking const &,
                                              LocalArgsBacking const &,
                                              TrainingLayerPlusContext const &,
                                              RealmRuntimeState &);

Future<std::optional<milliseconds_t>> execute_backward(LocalTaskRegistry const &,
                                               LocalTensorBacking const &,
                                               LocalArgsBacking const &,
                                               TrainingLayerPlusContext const &,
                                               RealmRuntimeState &);

Future<void> compute_loss(LocalTrainingBacking const &, LossAttrs const &, RealmRuntimeState &);

Future<void> execute_update(LocalTrainingBacking const &,
                    layer_guid_t const &,
                    OptimizerAttrs const &,
                    RealmRuntimeState &);

} // namespace FlexFlow

#endif
