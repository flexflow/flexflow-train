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
#include "task-spec/dynamic_graph/dynamic_tensor_accessor.dtg.h"
#include "task-spec/ff_iteration_config.dtg.h"
#include "utils/units/milliseconds_t.h"
#include <optional>

namespace FlexFlow {

struct ComputationGraphInstance {
public:
  ComputationGraphInstance(Allocator &,
                           std::vector<DynamicNodeInvocation> const &,
                           OptimizerAttrs const &,
                           std::optional<LossAttrs> const &,
                           std::optional<GenericTensorAccessorW>);
  Allocator &get_allocator() const;
  std::vector<DynamicNodeInvocation> const &get_execution_order() const;
  OptimizerAttrs const &get_optimizer_attrs() const;
  void update_optimizer_attrs_for_next_iter();
  std::optional<LossAttrs> const &get_loss_attrs() const;
  std::optional<GenericTensorAccessorR> get_loss_tensor_accessor() const;

private:
  Allocator &allocator;
  std::vector<DynamicNodeInvocation> execution_order;
  OptimizerAttrs optimizer_attrs;
  std::optional<LossAttrs> loss_attrs;
  std::optional<GenericTensorAccessorW> logit_grad_tensor;
};

ComputationGraphInstance create_computation_graph_instance(
    ComputationGraph const &compgraph,
    OptimizerAttrs const &optimizer_attrs,
    std::optional<LossAttrs> const &loss_attrs,
    std::optional<GenericTensorAccessorR> label_tensor,
    std::optional<dynamic_tensor_guid_t> logit_tensor,
    std::unordered_map<DynamicValueAttrs, DynamicTensorAccessor> const
        &input_tensors,
    Allocator &allocator,
    ProfilingSettings const &profiling_settings,
    device_handle_t const &device_handle,
    FFIterationConfig const &iteration_config,
    device_id_t device_idx);

std::unordered_map<dynamic_layer_guid_t, std::optional<milliseconds_t>>
    perform_all_passes_for_computation_graph_instance(
        ComputationGraphInstance &,
        ProfilingSettings const &profiling_settings,
        device_handle_t const &ff_handle,
        FFIterationConfig iteration_config,
        device_id_t device_idx);
std::unordered_map<dynamic_layer_guid_t, std::optional<milliseconds_t>>
    perform_forward_pass_for_computation_graph_instance(
        ComputationGraphInstance const &,
        ProfilingSettings const &profiling_settings,
        device_handle_t const &ff_handle,
        FFIterationConfig iteration_config,
        device_id_t device_idx);
std::unordered_map<dynamic_layer_guid_t, std::optional<milliseconds_t>>
    perform_backward_pass_for_computation_graph_instance(
        ComputationGraphInstance const &,
        ProfilingSettings const &profiling_settings,
        device_handle_t const &ff_handle,
        FFIterationConfig iteration_config,
        device_id_t device_idx);
void perform_update_pass_for_computation_graph_instance(
    ComputationGraphInstance &,
    ProfilingSettings const &profiling_settings,
    device_handle_t const &ff_handle,
    FFIterationConfig iteration_config,
    device_id_t device_idx);

} // namespace FlexFlow

#endif
