#include "internal/test_utils.h"
#include "kernels/local_cuda_allocator.h"
#include "kernels/managed_ff_stream.h"
#include "kernels/managed_per_device_ff_handle.h"
#include "local-execution/local_training_backing.h"
#include "op-attrs/ops/loss_functions/loss_attrs.dtg.h"
#include "pcg/computation_graph.h"
#include "pcg/computation_graph_builder.h"
#include "pcg/optimizer_attrs.dtg.h"
#include "task-spec/forward_tensor_source.h"
#include "task-spec/gradient_tensor_source.h"
#include "task-spec/loss_tensor_source.h"
#include "task-spec/optimizer_tensor_source.h"
#include "task-spec/runtime_arg_config.h"
#include "task-spec/training_computation_graph.h"
#include "utils/containers/get_only.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_CUDA_TEST_SUITE) {
  TEST_CASE("LossFunctions") {
    // initialize runtime
    ManagedFFStream managed_stream{};
    ManagedPerDeviceFFHandle managed_handle = initialize_single_gpu_handle(
        /*workSpaceSize=*/1024 * 1024,
        /*allowTensorOpMathConversion=*/true);

    Allocator allocator = create_local_cuda_memory_allocator();

    positive_int batch_size = 10_p;
    positive_int data_dim = 16_p;
    positive_int output_dim = 32_p;

    // construct computation graph
    ComputationGraph computation_graph = make_empty_computation_graph();

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
                   std::nullopt},
        {},
        {});

    LayerAddedResult linear_operator = add_layer(
        computation_graph,
        LayerAttrs{ComputationGraphOpAttrs{LinearAttrs{output_dim,
                                                       /*use_bias=*/false,
                                                       DataType::FLOAT,
                                                       Activation::RELU,
                                                       std::nullopt}},
                   std::nullopt},
        inputs_layer.outputs,
        weights_layer.outputs);
    tensor_guid_t logit_tensor = get_only(linear_operator.outputs);

    RuntimeArgConfig runtime_arg_config = gpu_make_runtime_arg_config(
        managed_handle.raw_handle(),
        EnableProfiling::YES,
        ProfilingSettings{/*warmup_iters=*/0, /*measure_iters=*/1});

    OptimizerAttrs optimizer_attrs = OptimizerAttrs{
        SGDOptimizerAttrs{
            /*lr=*/0.0,
            /*momentum=*/0.0,
            /*nesterov=*/false,
            /*weight_decay=*/0.0,
        },
    };

    ForwardTensorSource forward_tensor_source;
    GradientTensorSource gradient_tensor_source;
    OptimizerTensorSource optimizer_tensor_source;
    LossTensorSource loss_tensor_source;

    TrainingComputationGraph training_computation_graph =
        generate_training_computation_graph(computation_graph,
                                            optimizer_attrs,
                                            logit_tensor,
                                            forward_tensor_source,
                                            gradient_tensor_source,
                                            optimizer_tensor_source,
                                            loss_tensor_source);

    auto make_training_backing = [&](TensorShape const &label_tensor_shape) {
      GenericTensorAccessorW label_tensor_accessor =
          allocator.allocate_tensor(label_tensor_shape);

      return make_local_training_backing_for_computation_graph(
          /*allocator=*/allocator,
          /*preallocated_tensors=*/
          {
              {
                  training_tensor_guid_t{
                      training_computation_graph.label_tensor},
                  label_tensor_accessor,
              },
          },
          /*training_computation_graph=*/training_computation_graph,
          /*runtime_arg_config=*/runtime_arg_config,
          /*optimizer_attrs=*/optimizer_attrs);
    };

    SUBCASE("SparseCategoricalCrossEntropyLossAttrs") {
      TensorShape label_tensor_shape =
          TensorShape{TensorDims{FFOrdered{batch_size, 1_p}}, DataType::FLOAT};

      LocalTrainingBacking local_training_backing =
          make_training_backing(label_tensor_shape);

      LossAttrs loss_attrs = LossAttrs{
          SparseCategoricalCrossEntropyLossAttrs{/*replace_labels=*/false}};

      compute_loss(local_training_backing, loss_attrs, allocator);
    }

    SUBCASE("NonconfigurableLossAttrs") {
      TensorShape label_tensor_shape = TensorShape{
          TensorDims{FFOrdered{batch_size, output_dim}}, DataType::FLOAT};

      LocalTrainingBacking local_training_backing =
          make_training_backing(label_tensor_shape);

      SUBCASE("LossFunction::CATEGORICAL_CROSSENTROPY") {
        LossAttrs loss_attrs = LossAttrs{
            NonconfigurableLossAttrs{LossFunction::CATEGORICAL_CROSSENTROPY}};

        compute_loss(local_training_backing, loss_attrs, allocator);
      }

      SUBCASE("LossFunction::MEAN_SQUARED_ERROR_AVG_REDUCE") {
        LossAttrs loss_attrs = LossAttrs{NonconfigurableLossAttrs{
            LossFunction::MEAN_SQUARED_ERROR_AVG_REDUCE}};

        compute_loss(local_training_backing, loss_attrs, allocator);
      }

      SUBCASE("LossFunction::IDENTITY") {
        LossAttrs loss_attrs =
            LossAttrs{NonconfigurableLossAttrs{LossFunction::IDENTITY}};

        compute_loss(local_training_backing, loss_attrs, allocator);
      }
    }
  }
}
