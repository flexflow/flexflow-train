#ifndef _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_PARALLEL_COMPUTATION_GRAPH_INSTANCE_H
#define _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_PARALLEL_COMPUTATION_GRAPH_INSTANCE_H

#include "kernels/accessor.h"
#include "kernels/allocation.h"
#include "kernels/device_handle_t.dtg.h"
#include "kernels/profiling_settings.dtg.h"
#include "op-attrs/ops/loss_functions/loss_attrs.dtg.h"
#include "pcg/device_id_t.dtg.h"
#include "pcg/mapped_parallel_computation_graph/mapped_parallel_computation_graph.dtg.h"
#include "pcg/optimizer_attrs.dtg.h"
#include "realm-execution/realm_context.h"
#include "task-spec/dynamic_graph/dynamic_open_dataflow_graph.dtg.h"
#include "task-spec/dynamic_graph/dynamic_tensor_accessor.dtg.h"
#include "task-spec/dynamic_graph/dynamic_value_attrs.dtg.h"
#include "task-spec/ff_iteration_config.dtg.h"
#include "utils/units/milliseconds_t.h"
#include <optional>

namespace FlexFlow {

struct ParallelComputationGraphInstance {
public:
  ParallelComputationGraphInstance(RealmContext &,
                                   DynamicOpenDataflowGraph,
                                   std::vector<DynamicNodeInvocation> const &,
                                   OptimizerAttrs const &,
                                   std::optional<LossAttrs> const &,
                                   std::optional<GenericTensorAccessorW>);
  DynamicOpenDataflowGraph const &get_dynamic_dataflow_graph() const;
  Allocator &get_allocator() const;
  std::vector<DynamicNodeInvocation> const &get_topological_ordering() const;
  OptimizerAttrs const &get_optimizer_attrs() const;
  void update_optimizer_attrs_for_next_iter();
  std::optional<LossAttrs> const &get_loss_attrs() const;
  std::optional<GenericTensorAccessorR> get_loss_tensor_accessor() const;

private:
  RealmContext &realm;
  DynamicOpenDataflowGraph dataflow_graph;
  std::vector<DynamicNodeInvocation> topological_ordering;
  OptimizerAttrs optimizer_attrs;
  std::optional<LossAttrs> loss_attrs;
  std::optional<GenericTensorAccessorW> logit_grad_tensor;
};

ParallelComputationGraphInstance create_parallel_computation_graph_instance(
    RealmContext &realm,
    MappedParallelComputationGraph const &mpcg,
    OptimizerAttrs const &optimizer_attrs,
    std::optional<LossAttrs> const &loss_attrs,
    std::optional<GenericTensorAccessorR> label_tensor,
    std::optional<dynamic_tensor_guid_t> logit_tensor,
    std::unordered_map<DynamicValueAttrs, DynamicTensorAccessor> const
        &input_tensors,
    ProfilingSettings const &profiling_settings,
    FFIterationConfig const &iteration_config);

} // namespace FlexFlow

#endif
