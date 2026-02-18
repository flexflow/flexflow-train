#ifndef _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_DISTRIBUTED_PER_DEVICE_OP_STATE_BACKING_H
#define _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_DISTRIBUTED_PER_DEVICE_OP_STATE_BACKING_H

#include "kernels/profiling_settings.dtg.h"
#include "pcg/optimizer_attrs.dtg.h"
#include "realm-execution/distributed_device_handle.h"
#include "realm-execution/per_device_op_state_backing.dtg.h"
#include "realm-execution/realm_context.h"
#include "realm-execution/tensor_instance_backing.dtg.h"
#include "task-spec/dynamic_graph/dynamic_open_dataflow_graph.dtg.h"
#include "task-spec/ff_iteration_config.dtg.h"

namespace FlexFlow {

PerDeviceOpStateBacking perform_distributed_device_state_initialization(
    RealmContext &ctx,
    DynamicOpenDataflowGraph const &dg,
    TensorInstanceBacking const &tensor_instance_backing,
    ProfilingSettings const &profiling_settings,
    DistributedDeviceHandle const &device_handle,
    FFIterationConfig const &iteration_config,
    OptimizerAttrs const &optimizer_attrs,
    Realm::Event precondition);

} // namespace FlexFlow

#endif
