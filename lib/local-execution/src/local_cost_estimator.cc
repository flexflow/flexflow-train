#include "local-execution/local_cost_estimator.h"
#include "kernels/device.h"
#include "kernels/local_cuda_allocator.h"
#include "local-execution/tracked_allocator.h"
#include "op-attrs/computation_graph_op_attrs.h"
#include "op-attrs/pcg_operator_attrs.h"
#include "pcg/computation_graph.h"
#include "pcg/computation_graph/layer_added_result.dtg.h"
#include "pcg/machine_view.dtg.h"
#include "pcg/parallel_tensor_attrs.h"
#include "utils/containers/concat_vectors.h"
#include "utils/containers/get_only.h"
#include "utils/containers/sum.h"
#include "utils/containers/transform.h"
#include "utils/containers/values.h"

namespace FlexFlow {

LocalCostEstimator::LocalCostEstimator(RuntimeArgConfig const &config)
    : runtime_arg_config(config) {}

static ComputationGraph create_computation_graph_for_local_cost_estimation(
    PCGOperatorAttrs const &op,
    std::vector<ParallelTensorShape> const &inputs,
    std::vector<ParallelTensorAttrs> const &weights,
    std::vector<ParallelTensorAttrs> const &outputs) {
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
  for (ParallelTensorAttrs const &weight : weights) {
    LayerAddedResult weights_layer =
        add_layer(computation_graph,
                  LayerAttrs{ComputationGraphOpAttrs{WeightAttrs{
                                 get_piece_shape(weight.shape),
                                 InitializerAttrs{ZeroInitializerAttrs{}}}},
                             std::nullopt},
                  {},
                  {});
    weight_tensors.push_back(get_only(weights_layer.outputs));
  }

  // create operator layer
  LayerAddedResult operator_layer = add_layer(
      computation_graph,
      LayerAttrs{compgraph_op_attrs_from_pcg_op_attrs(op), "operator"},
      input_tensors,
      weight_tensors);

  return computation_graph;
}

CostDetails LocalCostEstimator::estimate_cost(
    PCGOperatorAttrs const &op,
    std::vector<ParallelTensorShape> const &inputs,
    std::vector<ParallelTensorAttrs> const &weights,
    std::vector<ParallelTensorAttrs> const &outputs,
    MachineView const &mv) const {

  if (is_parallel_op(op) || op.has<InputAttrs>() || op.has<NoopAttrs>() ||
      op.has<WeightAttrs>()) {
    return CostDetails{0, 0};
  }

  // construct computation graph
  ComputationGraph computation_graph =
      create_computation_graph_for_local_cost_estimation(
          op, inputs, weights, outputs);

  // allocate memory
  std::shared_ptr<TrackedAllocator> tracked_allocator_ptr =
      std::make_shared<TrackedAllocator>(create_local_cuda_memory_allocator());
  Allocator allocator = Allocator(tracked_allocator_ptr);

  GradientTensorSource gradient_tensor_source;

  LocalTrainingBacking local_backing(allocator,
                                     AllocatedTensors{{}, {}, {}},
                                     gradient_tensor_source,
                                     computation_graph,
                                     this->runtime_arg_config);
  // execute layer
  layer_guid_t operator_layer_guid = get_layer_by_name(computation_graph, "operator");
  
  float fwd = execute_forward(local_backing, operator_layer_guid, allocator).value();
  std::cout << "completed forward" << std::endl;
  float bwd = execute_backward(local_backing, operator_layer_guid, allocator).value();
  std::cout << "completed  backward" << std::endl;

  float total_execution_time = fwd + bwd;

  return CostDetails{total_execution_time,
                     tracked_allocator_ptr->get_current_mem_usage()};
}

float LocalCostEstimator::estimate_cost(ParallelTensorShape const &tensor_shape,
                                        MachineView const &src,
                                        MachineView const &dst) const {
  // TODO: model communication cost analytically
  // https://github.com/flexflow/FlexFlow/issues/1414
  // temporarily return 0

  return 0.0;
}

CostEstimator
    get_local_cost_estimator(RuntimeArgConfig const &runtime_arg_config) {
  return CostEstimator::create<LocalCostEstimator>(runtime_arg_config);
}

} // namespace FlexFlow
