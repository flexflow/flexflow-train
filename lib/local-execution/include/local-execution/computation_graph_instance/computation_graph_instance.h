#ifndef _FLEXFLOW_LIB_LOCAL_EXECUTION_INCLUDE_LOCAL_EXECUTION_COMPUTATION_GRAPH_INSTANCE_H
#define _FLEXFLOW_LIB_LOCAL_EXECUTION_INCLUDE_LOCAL_EXECUTION_COMPUTATION_GRAPH_INSTANCE_H

#include "kernels/device_handle_t.dtg.h"
#include "kernels/profiling_settings.dtg.h"
#include "op-attrs/ops/loss_functions/loss_attrs.dtg.h"
#include "pcg/computation_graph.dtg.h"
#include "pcg/device_id_t.dtg.h"
#include "pcg/optimizer_attrs.dtg.h"
#include "task-spec/dynamic_graph/dynamic_layer_guid_t.dtg.h"
#include "task-spec/dynamic_graph/dynamic_open_dataflow_graph.dtg.h"
#include "task-spec/ff_iteration_config.dtg.h"
#include "utils/units/milliseconds_t.h"

namespace FlexFlow {

struct ComputationGraphInstance {
public:
  ComputationGraphInstance(DynamicOpenDataflowGraph,
                           Allocator &,
                           std::vector<DynamicNodeInvocation> const &);
  DynamicOpenDataflowGraph const &get_dynamic_dataflow_graph() const;
  Allocator &get_allocator() const;
  std::vector<DynamicNodeInvocation> const &get_topological_ordering() const;

private:
  DynamicOpenDataflowGraph dataflow_graph;
  Allocator &allocator;
  std::vector<DynamicNodeInvocation> topological_ordering;
};

ComputationGraphInstance create_computation_graph_instance(
    ComputationGraph const &,
    OptimizerAttrs const &,
    std::unordered_map<DynamicValueAttrs, DynamicTensorAccessor> const &,
    Allocator &,
    ProfilingSettings const &,
    device_handle_t const &,
    FFIterationConfig const &,
    device_id_t);

std::unordered_map<dynamic_layer_guid_t, std::optional<milliseconds_t>>
    perform_all_passes_for_computation_graph_instance(
        ComputationGraphInstance const &,
        ProfilingSettings const &,
        device_handle_t const &,
        std::optional<LossAttrs> const &,
        FFIterationConfig,
        std::optional<OptimizerAttrs> const &,
        device_id_t);
std::unordered_map<dynamic_layer_guid_t, std::optional<milliseconds_t>>
    perform_forward_pass_for_computation_graph_instance(
        ComputationGraphInstance const &);
std::unordered_map<dynamic_layer_guid_t, std::optional<milliseconds_t>>
    perform_backward_pass_for_computation_graph_instance(
        ComputationGraphInstance const &);
void perform_update_pass_for_computation_graph_instance(
    ComputationGraphInstance const &);

} // namespace FlexFlow

#endif
