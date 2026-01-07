#include "local-execution/cost_estimator/local_cost_estimator.h"
#include "compiler/machine_mapping/machine_view.dtg.h"
#include "kernels/create_local_allocator_for_device_type.h"
#include "kernels/device.h"
#include "kernels/local_cpu_allocator.h"
#include "kernels/local_cuda_allocator.h"
#include "local-execution/cost_estimator/tracked_allocator.h"
#include "op-attrs/computation_graph_op_attrs.h"
#include "op-attrs/pcg_operator_attrs.h"
#include "pcg/computation_graph.h"
#include "pcg/computation_graph/layer_added_result.dtg.h"
#include "pcg/parallel_tensor_attrs.h"
#include "utils/containers/concat_vectors.h"
#include "utils/containers/get_only.h"
#include "utils/containers/maximum.h"
#include "utils/containers/sum.h"
#include "utils/containers/transform.h"
#include "utils/containers/unordered_set_of.h"
#include "utils/containers/values.h"

namespace FlexFlow {

LocalCostEstimator::LocalCostEstimator(RuntimeArgConfig const &config)
    : runtime_arg_config(config) {}

static ComputationGraph computation_graph_for_local_cost_estimation(
    ComputationGraphOpAttrs const &op,
    std::vector<ParallelTensorShape> const &inputs,
    std::vector<ParallelTensorShape> const &weights,
    std::vector<ParallelTensorShape> const &outputs) {
  ComputationGraph computation_graph = make_empty_computation_graph();

  std::vector<tensor_guid_t> input_tensors;
  for (ParallelTensorShape const &input : inputs) {
    LayerAddedResult inputs_layer = add_layer(
        computation_graph,
        LayerAttrs{ComputationGraphOpAttrs{InputAttrs{get_piece_shape(input)}},
                   std::nullopt},
        {},
        {});
    input_tensors.push_back(get_only(inputs_layer.outputs));
  }

  std::vector<tensor_guid_t> weight_tensors;
  for (ParallelTensorShape const &weight : weights) {
    LayerAddedResult weights_layer =
        add_layer(computation_graph,
                  LayerAttrs{ComputationGraphOpAttrs{WeightAttrs{
                                 get_piece_shape(weight),
                                 InitializerAttrs{ZeroInitializerAttrs{}}}},
                             std::nullopt},
                  {},
                  {});
    weight_tensors.push_back(get_only(weights_layer.outputs));
  }

  // create operator layer
  LayerAddedResult operator_layer = add_layer(computation_graph,
                                              LayerAttrs{
                                                  /*op_attrs=*/op,
                                                  /*name=*/"operator",
                                              },
                                              input_tensors,
                                              weight_tensors);

  return computation_graph;
}

OpCostMetrics LocalCostEstimator::estimate_cost(
    OpCostEstimateKey const &op_cost_estimate_key) const {

  PCGOperatorAttrs op = op_cost_estimate_key.op_attrs;
  std::vector<ParallelTensorShape> inputs = op_cost_estimate_key.input_shapes;
  std::vector<ParallelTensorShape> weights = op_cost_estimate_key.weight_shapes;
  std::vector<ParallelTensorShape> outputs = op_cost_estimate_key.output_shapes;
  MachineView mv = op_cost_estimate_key.machine_view;

  if (is_parallel_op(op) || op.has<InputAttrs>() || op.has<NoopAttrs>() ||
      op.has<WeightAttrs>()) {
    return OpCostMetrics{
        /*forward_runtime=*/0_ms,
        /*backward_runtime=*/0_ms,
        /*memory=*/0_bytes,
    };
  }

  // allocate memory
  std::shared_ptr<TrackedAllocator> tracked_allocator_ptr =
      std::make_shared<TrackedAllocator>(create_local_allocator_for_device_type(
          runtime_arg_config.kernel_device_type));

  layer_guid_t layer_guid = layer_guid_t{Node{0}};

  Allocator allocator = Allocator(tracked_allocator_ptr);

  // execute layer
  layer_guid_t operator_layer_guid =
      get_layer_by_name(training_cg.computation_graph, "operator");

  milliseconds_t fwd = execute_forward(local_backing.local_task_registry,
                                       local_backing.local_tensor_backing,
                                       local_backing.local_args_backing,
                                       get_training_layer_plus_context(
                                           training_cg, operator_layer_guid),
                                       allocator)
                           .value();
  milliseconds_t bwd = execute_backward(local_backing.local_task_registry,
                                        local_backing.local_tensor_backing,
                                        local_backing.local_args_backing,
                                        get_training_layer_plus_context(
                                            training_cg, operator_layer_guid),
                                        allocator)
                           .value();

  return OpCostMetrics{
      /*forward_runtime=*/fwd,
      /*backward_runtime=*/bwd,
      /*memory=*/tracked_allocator_ptr->get_current_mem_usage(),
  };
}

milliseconds_t LocalCostEstimator::estimate_cost(
    TensorSetMovement const &tensor_set_movement) const {

  auto estimate_single_comm_cost =
      [&](MachineSpaceCoordinate const &src,
          MachineSpaceCoordinate const &dst,
          num_bytes_t num_bytes) -> milliseconds_t {
    if (src == dst) {
      return 0_ms;
    } else if (src.node_idx == dst.node_idx) {
      return (num_bytes /
              this->interconnect_specification.intra_node_bandwidth);
    } else {
      return (num_bytes /
              this->interconnect_specification.inter_node_bandwidth);
    }
  };

  return maximum(
      transform(unordered_set_of(tensor_set_movement.edge_to_size),
                [&](std::pair<CommunicationEdge, num_bytes_t> const &p) {
                  return estimate_single_comm_cost(
                      p.first.get_src(), p.first.get_dst(), p.second);
                }));
}

CostEstimator
    get_local_cost_estimator(RuntimeArgConfig const &runtime_arg_config) {
  return CostEstimator::create<LocalCostEstimator>(runtime_arg_config);
}

} // namespace FlexFlow
