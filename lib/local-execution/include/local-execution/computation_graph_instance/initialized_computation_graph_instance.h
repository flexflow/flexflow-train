#ifndef _FLEXFLOW_LIB_LOCAL_EXECUTION_INCLUDE_LOCAL_EXECUTION_INITIALIZED_COMPUTATION_GRAPH_INSTANCE_H
#define _FLEXFLOW_LIB_LOCAL_EXECUTION_INCLUDE_LOCAL_EXECUTION_INITIALIZED_COMPUTATION_GRAPH_INSTANCE_H

#include "local-execution/computation_graph_instance/computation_graph_instance.h"
#include "local-execution/local_task_registry.dtg.h"
#include "pcg/layer_guid_t.dtg.h"
#include "task-spec/dynamic_graph/dynamic_open_dataflow_graph.dtg.h"
#include "utils/units/milliseconds_t.h"

namespace FlexFlow {

struct InitializedComputationGraphInstance {
public:
  LocalTaskRegistry const &get_task_registry() const;
  DynamicOpenDataflowGraph const &get_dynamic_dataflow_graph() const;
  Allocator &get_allocator() const;

private:
  Allocator &allocator;
  DynamicOpenDataflowGraph initialized_dataflow_graph;
};

InitializedComputationGraphInstance initialize_computation_graph_instance(
    ComputationGraphInstance const &,
    // FIXME (Elliott): figure out the right type to go here
    bidict<tensor_guid_t,
           std::variant<GenericTensorAccessorW, GenericTensorAccessorR>> const
        &,
    Allocator &);

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
