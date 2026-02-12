#ifndef _FLEXFLOW_LIB_LOCAL_EXECUTION_INCLUDE_LOCAL_EXECUTION_DEVICE_STATE_INITIALIZATION_H
#define _FLEXFLOW_LIB_LOCAL_EXECUTION_INCLUDE_LOCAL_EXECUTION_DEVICE_STATE_INITIALIZATION_H

#include "kernels/allocation.h"
#include "kernels/device_handle_t.dtg.h"
#include "kernels/profiling_settings.dtg.h"
#include "pcg/device_id_t.dtg.h"
#include "pcg/optimizer_attrs.dtg.h"
#include "task-spec/dynamic_graph/dynamic_open_dataflow_graph.dtg.h"
#include "task-spec/ff_iteration_config.dtg.h"

namespace FlexFlow {

bool no_nodes_are_initialized(DynamicOpenDataflowGraph const &g);

DynamicNodeInvocation
    initialize_node(DynamicNodeInvocation const &i,
                    Allocator &allocator,
                    ProfilingSettings const &profiling_settings,
                    device_handle_t const &device_handle,
                    FFIterationConfig const &iteration_config,
                    OptimizerAttrs const &optimizer_attrs,
                    device_id_t device_idx);

/**
 * @brief Initialize all operators and save the per-device op state
 */
DynamicOpenDataflowGraph perform_device_state_initialization(
    DynamicOpenDataflowGraph const &,
    Allocator &allocator,
    ProfilingSettings const &profiling_settings,
    device_handle_t const &device_handle,
    FFIterationConfig const &iteration_config,
    OptimizerAttrs const &optimizer_attrs,
    device_id_t device_idx);

} // namespace FlexFlow

#endif
