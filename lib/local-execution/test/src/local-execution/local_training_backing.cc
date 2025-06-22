#include <doctest/doctest.h>
#include "local-execution/local_training_backing.h"
#include "kernels/local_cuda_allocator.h"
#include "kernels/managed_ff_stream.h"
#include "kernels/managed_per_device_ff_handle.h"
#include "pcg/computation_graph.h"
#include "pcg/computation_graph_builder.h"
#include "pcg/optimizer_attrs.dtg.h"
#include "internal/test_utils.h"
#include "task-spec/forward_tensor_source.h"
#include "task-spec/gradient_tensor_source.h"
#include "task-spec/optimizer_tensor_source.h"
#include "task-spec/training_computation_graph.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_CUDA_TEST_SUITE) {
  TEST_CASE("execute_update") {
    // initialize runtime configs
    ManagedFFStream managed_stream{};
    ManagedPerDeviceFFHandle managed_handle = initialize_single_gpu_handle(
        /*workSpaceSize=*/1024 * 1024,
        /*allowTensorOpMathConversion=*/true);

    Allocator allocator = create_local_cuda_memory_allocator();

    // construct computation graph
    ComputationGraph computation_graph = make_empty_computation_graph();

    positive_int batch_size = 10_p;
    positive_int data_dim = 16_p;
    positive_int output_dim = 32_p;

    TensorShape input_tensor_shape = TensorShape{
        TensorDims{FFOrdered{batch_size, data_dim}}, DataType::FLOAT};

    TensorShape weight_shape = TensorShape{
        TensorDims{FFOrdered{data_dim, output_dim}}, DataType::FLOAT};

    LayerAddedResult inputs_layer =
        add_input_layer(computation_graph, input_tensor_shape);

    LayerAddedResult weights_layer = add_layer(
        computation_graph,
        LayerAttrs{ComputationGraphOpAttrs{WeightAttrs{
                       weight_shape, InitializerAttrs{ZeroInitializerAttrs{}}}},
                   "weights"},
        {},
        {});

    LayerAddedResult linear_operator = add_layer(
        computation_graph,
        LayerAttrs{ComputationGraphOpAttrs{LinearAttrs{output_dim,
                                                       /*use_bias=*/false,
                                                       DataType::FLOAT,
                                                       Activation::RELU,
                                                       std::nullopt}},
                   "linear"},
        inputs_layer.outputs,
        weights_layer.outputs);

    RuntimeArgConfig runtime_arg_config = RuntimeArgConfig{
        DeviceSpecific<PerDeviceFFHandle>::create(managed_handle.raw_handle()),
        EnableProfiling::YES,
        ProfilingSettings{/*warmup_iters=*/0, /*measure_iters=*/1},
        DeviceType::GPU};

    ForwardTensorSource forward_tensor_source;
    GradientTensorSource gradient_tensor_source;
    OptimizerTensorSource optimizer_tensor_source;

    auto make_training_backing = [&](OptimizerAttrs const &optimizer_attrs) {
      TrainingComputationGraph training_computation_graph =
          generate_training_computation_graph(
            computation_graph,
            optimizer_attrs,
            forward_tensor_source,
            gradient_tensor_source,
            optimizer_tensor_source);

      return make_local_training_backing_for_computation_graph(
          /*allocator=*/allocator,
          /*preallocated_tensors=*/{},
          /*training_computation_graph=*/training_computation_graph,
          /*runtime_arg_config=*/runtime_arg_config,
          /*optimizer_attrs=*/optimizer_attrs);
    };

    SUBCASE("SGDOptimizerAttrs") {
      SUBCASE("momentum=0") {
        OptimizerAttrs optimizer_attrs =
            OptimizerAttrs{SGDOptimizerAttrs{/*lr=*/0.001,
                                             /*momentum=*/0.0f,
                                             /*nesterov=*/false,
                                             /*weight_decay=*/0.001}};

        execute_update(make_training_backing(optimizer_attrs),
                       linear_operator.layer,
                       optimizer_attrs,
                       allocator);
      }

      SUBCASE("momentum=0.9") {
        OptimizerAttrs optimizer_attrs =
            OptimizerAttrs{SGDOptimizerAttrs{/*lr=*/0.001,
                                             /*momentum=*/0.9,
                                             /*nesterov=*/false,
                                             /*weight_decay=*/0.001}};

        execute_update(make_training_backing(optimizer_attrs),
                       linear_operator.layer,
                       optimizer_attrs,
                       allocator);
      }
    }

    SUBCASE("AdamOptimizerAttrs") {
      OptimizerAttrs optimizer_attrs =
          OptimizerAttrs{AdamOptimizerAttrs{/*alpha=*/0.001,
                                            /*beta1=*/0.9,
                                            /*beta2=*/0.999,
                                            /*weight_decay=*/0.001,
                                            /*alpha_t=*/0.001,
                                            /*beta_t=*/0.9,
                                            /*beta2_t=*/0.999,
                                            /*epsilon=*/1e-8}};
      execute_update(make_training_backing(optimizer_attrs),
                     linear_operator.layer,
                     optimizer_attrs,
                     allocator);
    }
  }
}
