#include "local-execution/local_cost_estimator.h"
#include "kernels/device.h"
#include "kernels/local_cuda_allocator.h"
#include "local-execution/local_training_backing.h"
#include "local-execution/tracked_allocator.h"
#include "op-attrs/computation_graph_op_attrs.h"
#include "op-attrs/pcg_operator_attrs.h"
#include "pcg/computation_graph.h"
#include "pcg/computation_graph/layer_added_result.dtg.h"
#include "pcg/machine_view.dtg.h"
#include "pcg/parallel_tensor_attrs.h"
#include "task-spec/forward_tensor_source.h"
#include "task-spec/gradient_tensor_source.h"
#include "task-spec/optimizer_tensor_source.h"
#include "task-spec/training_computation_graph.h"
#include "utils/containers/concat_vectors.h"
#include "utils/containers/get_only.h"
#include "utils/containers/sum.h"
#include "utils/containers/transform.h"
#include "utils/containers/values.h"

namespace FlexFlow {

LocalCostEstimator::LocalCostEstimator(RuntimeArgConfig const &config,
                                       OptimizerAttrs const &optimizer_attrs)
    : runtime_arg_config(config), optimizer_attrs(optimizer_attrs) {}

static ComputationGraph create_computation_graph_for_local_cost_estimation(
    PCGOperatorAttrs const &op,
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
  LayerAddedResult operator_layer = add_layer(
      computation_graph,
      LayerAttrs{compgraph_op_attrs_from_pcg_op_attrs(op), "operator"},
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

  ComputationGraph computation_graph =
      create_computation_graph_for_local_cost_estimation(
          op, inputs, weights, outputs);

  ForwardTensorSource forward_tensor_source;
  GradientTensorSource gradient_tensor_source;
  OptimizerTensorSource optimizer_tensor_source;

  TrainingComputationGraph training_cg = generate_training_computation_graph(
      /*computation_graph=*/computation_graph,
      /*optimizer_attrs=*/this->optimizer_attrs,
      /*forward_tensor_source=*/forward_tensor_source,
      /*gradient_tensor_source=*/gradient_tensor_source,
      /*optimizer_tensor_source=*/optimizer_tensor_source);

  // allocate memory
  std::shared_ptr<TrackedAllocator> tracked_allocator_ptr =
      std::make_shared<TrackedAllocator>(create_local_cuda_memory_allocator());
  Allocator allocator = Allocator(tracked_allocator_ptr);

  LocalTrainingBacking local_backing =
      make_local_training_backing_for_computation_graph(
          /*allocator=*/allocator,
          /*preallocated_tensors=*/{},
          /*training_computation_graph=*/training_cg,
          /*runtime_arg_config=*/this->runtime_arg_config,
          /*optimizer_attrs=*/this->optimizer_attrs);

  // execute layer
  layer_guid_t operator_layer_guid =
      get_layer_by_name(computation_graph, "operator");

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
  // TODO: model communication cost analytically
  // https://github.com/flexflow/FlexFlow/issues/1414

  NOT_IMPLEMENTED();
}

CostEstimator
    get_local_cost_estimator(RuntimeArgConfig const &runtime_arg_config,
                             OptimizerAttrs const &optimizer_attrs) {
  return CostEstimator::create<LocalCostEstimator>(runtime_arg_config,
                                                   optimizer_attrs);
}

} // namespace FlexFlow
