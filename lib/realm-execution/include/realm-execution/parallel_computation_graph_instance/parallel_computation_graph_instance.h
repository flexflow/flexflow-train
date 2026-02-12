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
#include "pcg/parallel_computation_graph/parallel_tensor_guid_t.dtg.h"
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
  ParallelComputationGraphInstance(
      RealmContext &ctx,
      std::vector<DynamicNodeInvocation> const &execution_order,
      OptimizerAttrs const &optimizer_attrs,
      std::optional<Realm::RegionInstance> logit_grad_tensor);
  RealmContext &get_realm_context();
  std::vector<DynamicNodeInvocation> const &get_execution_order() const;
  OptimizerAttrs const &get_optimizer_attrs() const;
  void update_optimizer_attrs_for_next_iter();
  std::optional<Realm::RegionInstance> get_loss_tensor_instance() const;

private:
  RealmContext &ctx;
  std::vector<DynamicNodeInvocation> execution_order;
  OptimizerAttrs optimizer_attrs;
  std::optional<Realm::RegionInstance> logit_grad_tensor;
};

ParallelComputationGraphInstance create_parallel_computation_graph_instance(
    RealmContext &ctx,
    MappedParallelComputationGraph const &mpcg,
    OptimizerAttrs const &optimizer_attrs,
    std::optional<LossAttrs> const &loss_attrs,
    std::optional<GenericTensorAccessorR> label_tensor,
    std::optional<parallel_tensor_guid_t> logit_tensor,
    std::unordered_map<DynamicValueAttrs, DynamicTensorAccessor> const
        &input_tensors,
    ProfilingSettings const &profiling_settings,
    FFIterationConfig const &iteration_config);

} // namespace FlexFlow

#endif
