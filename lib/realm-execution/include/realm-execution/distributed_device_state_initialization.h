#ifndef _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_DISTRIBUTED_DEVICE_STATE_INITIALIZATION_H
#define _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_DISTRIBUTED_DEVICE_STATE_INITIALIZATION_H

#include "kernels/profiling_settings.dtg.h"
#include "pcg/optimizer_attrs.dtg.h"
#include "realm-execution/distributed_device_handle.h"
#include "realm-execution/realm_context.h"
#include "task-spec/dynamic_graph/dynamic_open_dataflow_graph.dtg.h"
#include "task-spec/ff_iteration_config.dtg.h"

namespace FlexFlow {

DynamicOpenDataflowGraph perform_distributed_device_state_initialization(
    DynamicOpenDataflowGraph const &dg,
    RealmContext &ctx,
    ProfilingSettings const &profiling_settings,
    DistributedDeviceHandle const &device_handle,
    FFIterationConfig const &iteration_config,
    OptimizerAttrs const &optimizer_attrs,
    Realm::Event precondition);

} // namespace FlexFlow

#endif
