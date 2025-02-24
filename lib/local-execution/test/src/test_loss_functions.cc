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

namespace FlexFlow {

TEST_SUITE(FF_CUDA_TEST_SUITE) {
  TEST_CASE("Loss Functions") {
    Allocator allocator = create_local_cuda_memory_allocator();

    // allocate label tensors
    LossTensorSource loss_tensor_source;
    loss_tensor_t label_for_nonconfigurable_loss_attrs =
        loss_tensor_source.new_loss_tensor();
    loss_tensor_t label_for_sparse_cce_loss_attrs =
        loss_tensor_source.new_loss_tensor();

    nonnegative_int batch_size = 10_n;
    nonnegative_int data_dim = 100_n;

    TensorShape input_tensor_shape = TensorShape{
        TensorDims{FFOrdered<nonnegative_int>{batch_size, data_dim}},
        DataType::FLOAT};
    TensorShape reduced_input_tensor_shape =
        TensorShape{TensorDims{FFOrdered<nonnegative_int>{batch_size, 1_n}},
                    DataType::FLOAT};

    GenericTensorAccessorW label_for_nonconfigurable_loss_attrs_backing =
        allocator.allocate_tensor(reduced_input_tensor_shape);
    GenericTensorAccessorW label_for_sparse_cce_loss_attrs_backing =
        allocator.allocate_tensor(reduced_input_tensor_shape);
    AllocatedTensors allocated_tensors = AllocatedTensors{
        {{TensorTypeVariant{label_for_nonconfigurable_loss_attrs},
          label_for_nonconfigurable_loss_attrs_backing},
         {TensorTypeVariant{label_for_sparse_cce_loss_attrs},
          label_for_sparse_cce_loss_attrs_backing}},
        {},
        {}};

    // construct computation graph
    ComputationGraph computation_graph = make_empty_computation_graph();

    TensorAttrs input_tensor_attrs = TensorAttrs{
        input_tensor_shape, std::nullopt, std::nullopt, CreateGrad::YES};

    LayerAddedResult inputs_layer =
        add_layer(computation_graph,
                  LayerAttrs{ComputationGraphOpAttrs{InputAttrs{}}, "inputs"},
                  {},
                  {input_tensor_attrs});

    float scalar = 4.0;
    LayerAddedResult scalar_multiply_operator =
        add_layer(computation_graph,
                  LayerAttrs{ComputationGraphOpAttrs{ElementUnaryAttrs{
                                 OperatorType::SCALAR_MULTIPLY, scalar}},
                             "scalar_mult"},
                  inputs_layer.outputs,
                  {input_tensor_attrs});
    tensor_guid_t label_tensor = scalar_multiply_operator.outputs.at(0);

    // initialize runtime configs
    ManagedPerDeviceFFHandle managed_handle{};

    RuntimeArgConfig runtime_arg_config = RuntimeArgConfig{
        DeviceSpecific<PerDeviceFFHandle>::create(managed_handle.raw_handle()),
        EnableProfiling::YES,
        ProfilingSettings{/*warmup_iters=*/0, /*measure_iters=*/1}};

    // initialize training backing
    LocalTrainingBacking local_training_backing = LocalTrainingBacking{
        allocator, allocated_tensors, computation_graph, runtime_arg_config};

    SUBCASE("SparseCategoricalCrossEntropyLossAttrs") {
      LossAttrs loss_attrs = LossAttrs{
          SparseCategoricalCrossEntropyLossAttrs{/*replace_labels=*/false}};

      compute_loss(local_training_backing,
                   loss_attrs,
                   label_tensor,
                   label_for_sparse_cce_loss_attrs,
                   allocator);
    }

    SUBCASE("NonconfigurableLossAttrs") {
      SUBCASE("LossFunction::CATEGORICAL_CROSSENTROPY") {
        LossAttrs loss_attrs = LossAttrs{
            NonconfigurableLossAttrs{LossFunction::CATEGORICAL_CROSSENTROPY}};
        compute_loss(local_training_backing,
                     loss_attrs,
                     label_tensor,
                     label_for_nonconfigurable_loss_attrs,
                     allocator);
      }

      SUBCASE("LossFunction::MEAN_SQUARED_ERROR_AVG_REDUCE") {
        LossAttrs loss_attrs = LossAttrs{NonconfigurableLossAttrs{
            LossFunction::MEAN_SQUARED_ERROR_AVG_REDUCE}};
        compute_loss(local_training_backing,
                     loss_attrs,
                     label_tensor,
                     label_for_nonconfigurable_loss_attrs,
                     allocator);
      }

      SUBCASE("LossFunction::IDENTITY") {
        LossAttrs loss_attrs =
            LossAttrs{NonconfigurableLossAttrs{LossFunction::IDENTITY}};
        compute_loss(local_training_backing,
                     loss_attrs,
                     label_tensor,
                     label_for_nonconfigurable_loss_attrs,
                     allocator);
      }
    }
  }
}

} // namespace FlexFlow
