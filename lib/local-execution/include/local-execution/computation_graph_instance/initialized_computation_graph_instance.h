#ifndef _FLEXFLOW_LIB_LOCAL_EXECUTION_INCLUDE_LOCAL_EXECUTION_INITIALIZED_COMPUTATION_GRAPH_INSTANCE_H
#define _FLEXFLOW_LIB_LOCAL_EXECUTION_INCLUDE_LOCAL_EXECUTION_INITIALIZED_COMPUTATION_GRAPH_INSTANCE_H

#include "kernels/device_handle_t.dtg.h"
#include "kernels/profiling_settings.dtg.h"
#include "pcg/computation_graph.dtg.h"
#include "pcg/layer_guid_t.dtg.h"
#include "pcg/optimizer_attrs.dtg.h"
#include "task-spec/dynamic_graph/dynamic_open_dataflow_graph.dtg.h"
#include "task-spec/ff_iteration_config.dtg.h"
#include "utils/units/milliseconds_t.h"

namespace FlexFlow {

struct InitializedComputationGraphInstance {
public:
  InitializedComputationGraphInstance(DynamicOpenDataflowGraph, Allocator &);
  DynamicOpenDataflowGraph const &get_dynamic_dataflow_graph() const;
  Allocator &get_allocator() const;

private:
  DynamicOpenDataflowGraph initialized_dataflow_graph;
  Allocator &allocator;
};

InitializedComputationGraphInstance initialize_computation_graph_instance(
    ComputationGraph const &cg,
    OptimizerAttrs const &optimizer,
    std::unordered_map<DynamicValueAttrs, DynamicTensorAccessor> const &,
    Allocator &,
    ProfilingSettings const &,
    device_handle_t const &,
    DeviceType,
    FFIterationConfig const &,
    size_t);

std::unordered_map<layer_guid_t, std::optional<milliseconds_t>>
    perform_forward_pass_for_computation_graph_instance(
        InitializedComputationGraphInstance const &);

std::unordered_map<layer_guid_t, std::optional<milliseconds_t>>
    perform_backward_pass_for_computation_graph_instance(
        InitializedComputationGraphInstance const &);

void perform_update_pass_for_computation_graph_instance(
    InitializedComputationGraphInstance const &);

} // namespace FlexFlow

#endif
