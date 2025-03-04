#ifndef _FLEXFLOW_LOCAL_EXECUTION_LOCAL_TRAINING_BACKING_H
#define _FLEXFLOW_LOCAL_EXECUTION_LOCAL_TRAINING_BACKING_H

#include "local-execution/local_slots_backing.h"
#include "local-execution/task_registry.h"
#include "pcg/computation_graph.dtg.h"

namespace FlexFlow {

using PerLayerElapsedTime =
    std::unordered_map<layer_guid_t, std::optional<float>>;

struct LocalTrainingBacking {
  LocalTrainingBacking(Allocator const &,
                       ComputationGraph const &,
                       TensorBackingMap const &,
                       RuntimeArgConfig const &);

  void execute_init();
  PerLayerElapsedTime execute_forward();
  PerLayerElapsedTime execute_backward();
  void execute_update();

  TaskArgumentAccessor get_task_arg_accessor(OpTaskInvocation const &,
                                             layer_guid_t const &) const;

private:
  DeviceSpecificDeviceStates call_init_task_impl(task_id_t,
                                                 TaskArgumentAccessor const &);
  std::optional<float> call_task_impl(task_id_t, TaskArgumentAccessor);

private:
  Allocator allocator;
  ComputationGraph computation_graph;
  TaskRegistry task_registry;
  LocalSlotsBacking local_slots_backing;
};

} // namespace FlexFlow

#endif
