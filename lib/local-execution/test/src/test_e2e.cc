#include "internal/test_utils.h"
#include "kernels/compare_tensor_accessors.h"
#include "kernels/copy_tensor_accessor.h"
#include "kernels/local_cpu_allocator.h"
#include "kernels/local_cuda_allocator.h"
#include "kernels/managed_ff_stream.h"
#include "kernels/managed_per_device_ff_handle.h"
#include "kernels/tensor_accessor_reductions.h"
#include "local-execution/local_training_backing.h"
#include "local-execution/model_training_instance.h"
#include "op-attrs/ops/loss_functions/loss_attrs.dtg.h"
#include "pcg/computation_graph.h"
#include "pcg/computation_graph_builder.h"
#include "pcg/optimizer_attrs.dtg.h"
#include "task-spec/forward_tensor_source.h"
#include "task-spec/gradient_tensor_source.h"
#include "task-spec/loss_tensor_source.h"
#include "task-spec/optimizer_tensor_source.h"
#include "task-spec/training_computation_graph.h"
#include "utils/containers/get_only.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

bool did_loss_decrease(GenericTensorAccessorR const &first_epoch,
                       GenericTensorAccessorR const &last_epoch) {
  Allocator cpu_allocator = create_local_cpu_memory_allocator();

  return tensor_accessor_all(
      compare_tensor_accessors_le(last_epoch, first_epoch, cpu_allocator));
}

TEST_SUITE(FF_CUDA_TEST_SUITE) {
  TEST_CASE("LocalBackend e2e Training") {
    // initialize runtime
    ManagedFFStream managed_stream{};
    ManagedPerDeviceFFHandle managed_handle = initialize_single_gpu_handle(
        /*workSpaceSize=*/1024 * 1024,
        /*allowTensorOpMathConversion=*/true);

    Allocator allocator = create_local_cuda_memory_allocator();

    // allocate label tensors
    LossTensorSource loss_tensor_source;
    loss_tensor_guid_t label_tensor_guid = loss_tensor_source.new_loss_tensor();

    positive_int batch_size = 10_p;
    positive_int data_dim = 16_p;
    positive_int hidden_dim = 32_p;
    positive_int output_dim = 1_p;

    TensorShape output_tensor_shape = TensorShape{
        TensorDims{FFOrdered{batch_size, output_dim}}, DataType::FLOAT};

    GenericTensorAccessorW label_tensor_backing =
        allocator.allocate_tensor(output_tensor_shape);

    // construct computation graph
    ComputationGraph computation_graph = make_empty_computation_graph();

    TensorShape input_tensor_shape = TensorShape{
        TensorDims{FFOrdered{batch_size, data_dim}}, DataType::FLOAT};

    TensorShape weight_shape_1 = TensorShape{
        TensorDims{FFOrdered{data_dim, hidden_dim}}, DataType::FLOAT};
    TensorShape weight_shape_2 = TensorShape{
        TensorDims{FFOrdered{hidden_dim, output_dim}}, DataType::FLOAT};

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
        ProfilingSettings{/*warmup_iters=*/0, /*measure_iters=*/1},
        DeviceType::GPU};

    // initialize training backing
    LossAttrs loss_attrs = LossAttrs{
        NonconfigurableLossAttrs{LossFunction::CATEGORICAL_CROSSENTROPY}};
    OptimizerAttrs optimizer_attrs =
        OptimizerAttrs{SGDOptimizerAttrs{/*lr=*/0.001,
                                         /*momentum=*/0.9,
                                         /*nesterov=*/false,
                                         /*weight_decay=*/0.001}};

    ForwardTensorSource forward_tensor_source;
    GradientTensorSource gradient_tensor_source;
    OptimizerTensorSource optimizer_tensor_source;

    TrainingComputationGraph training_computation_graph =
        generate_training_computation_graph(computation_graph,
                                            optimizer_attrs,
                                            forward_tensor_source,
                                            gradient_tensor_source,
                                            optimizer_tensor_source);

    LocalTrainingBacking local_training_backing =
        make_local_training_backing_for_computation_graph(
            /*allocator=*/allocator,
            /*preallocated_tensors=*/
            {
                {
                    training_tensor_guid_t{label_tensor_guid},
                    label_tensor_backing,
                },
            },
            /*training_computation_graph=*/training_computation_graph,
            /*runtime_arg_config=*/runtime_arg_config,
            /*optimizer_attrs=*/optimizer_attrs);

    // begin training loop
    ModelTrainingInstance model_training_instance =
        ModelTrainingInstance{allocator,
                              local_training_backing,
                              logit_tensor,
                              label_tensor_guid,
                              loss_attrs,
                              optimizer_attrs};

    Allocator cpu_allocator = create_local_cpu_memory_allocator();

    int num_epochs = 5;
    std::vector<GenericTensorAccessorR> loss_values;

    for (int i = 0; i < num_epochs; i++) {
      model_training_instance.forward();
      model_training_instance.backward();
      model_training_instance.update();
      loss_values.push_back(copy_tensor_accessor_r(
          model_training_instance.get_loss_tensor_accessor(), cpu_allocator));
    }

    // Assert that each sample in the batch has a lower loss in last epoch than
    // the first epoch
    GenericTensorAccessorR first_epoch_loss = loss_values.at(0);
    GenericTensorAccessorR last_epoch = loss_values.back();
    CHECK(did_loss_decrease(first_epoch_loss, last_epoch));
  }
}
