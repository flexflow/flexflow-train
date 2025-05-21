#include "doctest/doctest.h"
#include "kernels/local_cuda_allocator.h"
#include "kernels/managed_ff_stream.h"
#include "kernels/managed_per_device_ff_handle.h"
#include "local-execution/allocated_tensors.h"
#include "local-execution/local_training_backing.h"
#include "op-attrs/ops/loss_functions/loss_attrs.dtg.h"
#include "pcg/computation_graph.h"
#include "pcg/computation_graph_builder.h"
#include "pcg/optimizer_attrs.dtg.h"
#include "test_utils.h"
#include "utils/containers/get_only.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_CUDA_TEST_SUITE) {
  TEST_CASE("LossFunctions") {
    // initialize runtime
    ManagedFFStream managed_stream{};
    ManagedPerDeviceFFHandle managed_handle = initialize_single_gpu_handle(
      /*workSpaceSize=*/1024 * 1024,
      /*allowTensorOpMathConversion=*/true
    );

    Allocator allocator = create_local_cuda_memory_allocator();

    // allocate label tensors
    LossTensorSource loss_tensor_source;
    loss_tensor_t label_for_nonconfigurable_loss_attrs =
        loss_tensor_source.new_loss_tensor();
    loss_tensor_t label_for_sparse_cce_loss_attrs =
        loss_tensor_source.new_loss_tensor();

    positive_int batch_size = 10_p;
    positive_int data_dim = 16_p;
    positive_int output_dim = 32_p;

    TensorShape output_tensor_shape = TensorShape{
        TensorDims{FFOrdered{batch_size, output_dim}},
        DataType::FLOAT};
    TensorShape reduced_tensor_shape =
        TensorShape{TensorDims{FFOrdered{batch_size, 1_p}},
                    DataType::FLOAT};

    GenericTensorAccessorW label_for_nonconfigurable_loss_attrs_backing =
        allocator.allocate_tensor(output_tensor_shape);
    GenericTensorAccessorW label_for_sparse_cce_loss_attrs_backing =
        allocator.allocate_tensor(reduced_tensor_shape);
    AllocatedTensors allocated_tensors = AllocatedTensors{
        {{TensorTypeVariant{label_for_nonconfigurable_loss_attrs},
          label_for_nonconfigurable_loss_attrs_backing},
         {TensorTypeVariant{label_for_sparse_cce_loss_attrs},
          label_for_sparse_cce_loss_attrs_backing}},
        {},
        {}};

    // construct computation graph
    ComputationGraph computation_graph = make_empty_computation_graph();

    TensorShape input_tensor_shape = TensorShape{
        TensorDims{FFOrdered{batch_size, data_dim}},
        DataType::FLOAT};

    TensorShape weight_shape = TensorShape{
        TensorDims{FFOrdered{data_dim, output_dim}},
        DataType::FLOAT};

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

    RuntimeArgConfig runtime_arg_config = RuntimeArgConfig{
        DeviceSpecific<PerDeviceFFHandle>::create(managed_handle.raw_handle()),
        EnableProfiling::YES,
        ProfilingSettings{/*warmup_iters=*/0, /*measure_iters=*/1}};

    // initialize training backing
    GradientTensorSource gradient_tensor_source;
    LocalTrainingBacking local_training_backing =
        LocalTrainingBacking{allocator,
                             allocated_tensors,
                             gradient_tensor_source,
                             computation_graph,
                             runtime_arg_config};

    SUBCASE("SparseCategoricalCrossEntropyLossAttrs") {
      LossAttrs loss_attrs = LossAttrs{
          SparseCategoricalCrossEntropyLossAttrs{/*replace_labels=*/false}};

      compute_loss(local_training_backing,
                   loss_attrs,
                   logit_tensor,
                   label_for_sparse_cce_loss_attrs,
                   allocator);
    }

    SUBCASE("NonconfigurableLossAttrs") {
      SUBCASE("LossFunction::CATEGORICAL_CROSSENTROPY") {
        LossAttrs loss_attrs = LossAttrs{
            NonconfigurableLossAttrs{LossFunction::CATEGORICAL_CROSSENTROPY}};
        compute_loss(local_training_backing,
                     loss_attrs,
                     logit_tensor,
                     label_for_nonconfigurable_loss_attrs,
                     allocator);
      }

      SUBCASE("LossFunction::MEAN_SQUARED_ERROR_AVG_REDUCE") {
        LossAttrs loss_attrs = LossAttrs{NonconfigurableLossAttrs{
            LossFunction::MEAN_SQUARED_ERROR_AVG_REDUCE}};
        compute_loss(local_training_backing,
                     loss_attrs,
                     logit_tensor,
                     label_for_nonconfigurable_loss_attrs,
                     allocator);
      }

      SUBCASE("LossFunction::IDENTITY") {
        LossAttrs loss_attrs =
            LossAttrs{NonconfigurableLossAttrs{LossFunction::IDENTITY}};
        compute_loss(local_training_backing,
                     loss_attrs,
                     logit_tensor,
                     label_for_nonconfigurable_loss_attrs,
                     allocator);
      }
    }
  }
}
