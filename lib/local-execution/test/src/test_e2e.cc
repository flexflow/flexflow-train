#include "kernels/local_cuda_allocator.h"
#include "kernels/managed_ff_stream.h"
#include "kernels/managed_per_device_ff_handle.h"
#include "local-execution/allocated_tensors.h"
#include "local-execution/local_training_backing.h"
#include "local-execution/model_training_instance.h"
#include "op-attrs/ops/loss_functions/loss_attrs.dtg.h"
#include "pcg/computation_graph.h"
#include "pcg/computation_graph_builder.h"
#include "pcg/optimizer_attrs.dtg.h"
#include "test_utils.h"
#include "utils/containers/get_only.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

bool did_loss_decrease(float *first_epoch, float *last_epoch, int batch_size) {
  for (int i = 0; i < batch_size; i++) {
    if (first_epoch[i] < last_epoch[i]) {
      return false;
    }
  }
  return true;
}

TEST_SUITE(FF_CUDA_TEST_SUITE) {
  TEST_CASE("LocalBackend e2e Training") {
    // initialize runtime
    ManagedFFStream managed_stream{};
    ManagedPerDeviceFFHandle managed_handle = initialize_single_gpu_handle();

    Allocator allocator = create_local_cuda_memory_allocator();

    // allocate label tensors
    LossTensorSource loss_tensor_source;
    loss_tensor_t label_tensor = loss_tensor_source.new_loss_tensor();

    nonnegative_int batch_size = 10_n;
    nonnegative_int data_dim = 16_n;
    nonnegative_int hidden_dim = 32_n;
    nonnegative_int output_dim = 1_n;

    TensorShape output_tensor_shape = TensorShape{
        TensorDims{FFOrdered<nonnegative_int>{batch_size, output_dim}},
        DataType::FLOAT};

    GenericTensorAccessorW label_tensor_backing =
        allocator.allocate_tensor(output_tensor_shape);
    AllocatedTensors allocated_tensors = AllocatedTensors{
        /*tensor_type_backings=*/{
          {TensorTypeVariant{label_tensor}, label_tensor_backing},
        }, 
        /*gradient_mapping=*/{}, 
        /*optimizer_mapping*/{},
    };

    // construct computation graph
    ComputationGraph computation_graph = make_empty_computation_graph();

    TensorShape input_tensor_shape = TensorShape{
        TensorDims{FFOrdered<nonnegative_int>{batch_size, data_dim}},
        DataType::FLOAT};

    TensorShape weight_shape_1 = TensorShape{
        TensorDims{FFOrdered<nonnegative_int>{data_dim, hidden_dim}},
        DataType::FLOAT};
    TensorShape weight_shape_2 = TensorShape{
        TensorDims{FFOrdered<nonnegative_int>{hidden_dim, output_dim}},
        DataType::FLOAT};

    LayerAddedResult inputs_layer =
        add_input_layer_with_grad(computation_graph, input_tensor_shape);

    LayerAddedResult weights_layer_1 = add_layer(
        computation_graph,
        LayerAttrs{ComputationGraphOpAttrs{WeightAttrs{
                       weight_shape_1, InitializerAttrs{GlorotNormalAttrs{0}}}},
                   std::nullopt},
        {},
        {});

    LayerAddedResult weights_layer_2 = add_layer(
        computation_graph,
        LayerAttrs{ComputationGraphOpAttrs{WeightAttrs{
                       weight_shape_2, InitializerAttrs{GlorotNormalAttrs{0}}}},
                   std::nullopt},
        {},
        {});

    LayerAddedResult linear_operator_1 = add_layer(
        computation_graph,
        LayerAttrs{ComputationGraphOpAttrs{LinearAttrs{hidden_dim,
                                                       /*use_bias=*/false,
                                                       DataType::FLOAT,
                                                       Activation::RELU,
                                                       std::nullopt}},
                   std::nullopt},
        inputs_layer.outputs,
        weights_layer_1.outputs);

    LayerAddedResult linear_operator_2 = add_layer(
        computation_graph,
        LayerAttrs{ComputationGraphOpAttrs{LinearAttrs{output_dim,
                                                       /*use_bias=*/false,
                                                       DataType::FLOAT,
                                                       Activation::RELU,
                                                       std::nullopt}},
                   std::nullopt},
        linear_operator_1.outputs,
        weights_layer_2.outputs);

    tensor_guid_t logit_tensor = get_only(linear_operator_2.outputs);

    RuntimeArgConfig runtime_arg_config = RuntimeArgConfig{
        DeviceSpecific<PerDeviceFFHandle>::create(managed_handle.raw_handle()),
        EnableProfiling::YES,
        ProfilingSettings{/*warmup_iters=*/0, /*measure_iters=*/1}};

    // initialize training backing
    LossAttrs loss_attrs = LossAttrs{
        NonconfigurableLossAttrs{LossFunction::CATEGORICAL_CROSSENTROPY}};
    OptimizerAttrs optimizer_attrs =
        OptimizerAttrs{SGDOptimizerAttrs{/*lr=*/0.001,
                                         /*momentum=*/0.9,
                                         /*nesterov=*/false,
                                         /*weight_decay=*/0.001}};

    GradientTensorSource gradient_tensor_source;
    OptimizerTensorSource optimizer_tensor_source;

    LocalTrainingBacking local_training_backing =
        LocalTrainingBacking{allocator,
                             allocated_tensors,
                             gradient_tensor_source,
                             optimizer_tensor_source,
                             computation_graph,
                             runtime_arg_config,
                             optimizer_attrs};

    // begin training loop
    ModelTrainingInstance model_training_instance =
        ModelTrainingInstance{allocator,
                              local_training_backing,
                              logit_tensor,
                              label_tensor,
                              loss_attrs,
                              optimizer_attrs};

    int num_epochs = 5;
    int num_samples = batch_size.unwrap_nonnegative();
    std::vector<float *> loss_values(num_epochs);

    for (int i = 0; i < num_epochs; i++) {
      model_training_instance.forward();
      model_training_instance.backward();
      model_training_instance.update();
      float *host_loss_ptr = new float[num_samples];
      model_training_instance.write_loss_tensor_to_host(host_loss_ptr);
      loss_values[i] = host_loss_ptr;
    }

    // Assert that each sample in the batch has a lower loss in last epoch than
    // the first epoch
    float *first_epoch = loss_values[0];
    float *last_epoch = loss_values[num_epochs - 1];
    CHECK(did_loss_decrease(
        first_epoch, last_epoch, batch_size.unwrap_nonnegative()));

    for (int i = 0; i < num_epochs; i++) {
      delete[] loss_values[i];
    }
  }
}
