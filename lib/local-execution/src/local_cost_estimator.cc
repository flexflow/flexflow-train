#include "local-execution/local_cost_estimator.h"
#include "kernels/device.h"
#include "kernels/local_cuda_allocator.h"
#include "local-execution/tensor_lowering.h"
#include "local-execution/tracked_allocator.h"
#include "op-attrs/computation_graph_op_attrs.h"
#include "op-attrs/pcg_operator_attrs.h"
#include "pcg/computation_graph/layer_added_result.dtg.h"
#include "pcg/computation_graph_builder.h"
#include "pcg/parallel_tensor_attrs.h"
#include "utils/containers/concat_vectors.h"
#include "utils/containers/sum.h"
#include "utils/containers/transform.h"
#include "utils/containers/values.h"

namespace FlexFlow {

LocalCostEstimator::LocalCostEstimator(RuntimeArgConfig const &config)
    : runtime_arg_config(config) {}

static ComputationGraph const &
    create_computation_graph_for_local_cost_estimation(
        PCGOperatorAttrs const &op,
        std::vector<ParallelTensorShape> const &inputs,
        std::vector<ParallelTensorAttrs> const &weights,
        std::vector<ParallelTensorAttrs> const &outputs) {
  ComputationGraph computation_graph = make_empty_computation_graph();

  // create layer for inputs
  auto get_vector_piece_attrs_from_parallel_tensor_shape =
      [](std::vector<ParallelTensorShape> const &parallel_shapes) {
        return transform(parallel_shapes, [](ParallelTensorShape const &p) {
          return TensorAttrs{
              get_piece_shape(p), std::nullopt, std::nullopt, CreateGrad::YES};
        });
      };

  LayerAddedResult inputs_layer =
      add_layer(computation_graph,
                LayerAttrs{ComputationGraphOpAttrs{InputAttrs{}}, "inputs"},
                {},
                get_vector_piece_attrs_from_parallel_tensor_shape(inputs));

  // create layer for weights
  auto get_vector_piece_attrs_from_parallel_tensor_attrs =
      [](std::vector<ParallelTensorAttrs> const &parallel_attrs) {
        return transform(parallel_attrs, [](ParallelTensorAttrs const &p) {
          return get_piece_attrs(p);
        });
      };

  LayerAddedResult weights_layer =
      add_layer(computation_graph,
                LayerAttrs{ComputationGraphOpAttrs{InputAttrs{}}, "weights"},
                {},
                get_vector_piece_attrs_from_parallel_tensor_attrs(weights));

  // create operator layer
  LayerAddedResult operator_layer = add_layer(
      computation_graph,
      LayerAttrs{compgraph_op_attrs_from_pcg_op_attrs(op), "operator"},
      concat_vectors(inputs_layer.outputs, weights_layer.outputs),
      get_vector_piece_attrs_from_parallel_tensor_attrs(outputs));

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

  LocalTrainingBacking local_backing(
      allocator,
      computation_graph,
      LocalTensorBacking{},
      LocalArgsBacking{this->runtime_arg_config});

  allocate_all_computation_graph_tensors(local_backing.local_tensor_backing,
                                         local_backing.gradient_tensor_source,
                                         local_backing.computation_graph,
                                         local_backing.allocator);

  // execute layer
  layer_guid_t operator_layer_guid =
      get_layer_by_name(computation_graph, "operator");
  execute_init(local_backing, operator_layer_guid);
  float fwd = execute_forward(local_backing, operator_layer_guid).value();
  float bwd = execute_backward(local_backing, operator_layer_guid).value();

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
