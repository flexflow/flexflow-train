#include "kernels/device_handle_t.h"
#include "kernels/local_cuda_allocator.h"
#include "kernels/managed_ff_stream.h"
#include "kernels/managed_per_device_ff_handle.h"
#include "local-execution/computation_graph_instance/computation_graph_instance.h"
#include "op-attrs/ops/loss_functions/loss_attrs.dtg.h"
#include "pcg/computation_graph.h"
#include "pcg/computation_graph_builder.h"
#include "pcg/device_id_t.h"
#include "pcg/optimizer_attrs.dtg.h"
#include "utils/containers/require_only_key.h"
#include "utils/optional.h"
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

    TensorShape label_tensor_shape = TensorShape{
        TensorDims{FFOrdered{batch_size, output_dim}}, DataType::FLOAT};
    GenericTensorAccessorW label_tensor =
        allocator.allocate_tensor(label_tensor_shape);

    TensorShape weight_shape = TensorShape{
        TensorDims{FFOrdered{data_dim, output_dim}}, DataType::FLOAT};

    LayerAddedResult inputs_layer =
        add_input_layer(computation_graph, input_tensor_shape);
    tensor_guid_t inputs_tensor =
        require_only_key(inputs_layer.outputs, TensorSlotName::OUTPUT);

    LayerAddedResult weights_layer = add_layer(
        computation_graph,
        LayerAttrs{ComputationGraphOpAttrs{WeightAttrs{
                       weight_shape, InitializerAttrs{ZeroInitializerAttrs{}}}},
                   std::nullopt},
        {},
        {});
    tensor_guid_t weights_tensor =
        require_only_key(weights_layer.outputs, TensorSlotName::OUTPUT);

    LayerAddedResult linear_operator = add_layer(
        computation_graph,
        LayerAttrs{ComputationGraphOpAttrs{LinearAttrs{output_dim,
                                                       /*use_bias=*/false,
                                                       DataType::FLOAT,
                                                       Activation::RELU,
                                                       std::nullopt}},
                   std::nullopt},
        {
            {
                TensorSlotName::INPUT,
                inputs_tensor,
            },
        },
        {
            {
                TensorSlotName::WEIGHT,
                weights_tensor,
            },
        });
    tensor_guid_t logit_tensor =
        require_only_key(linear_operator.outputs, TensorSlotName::OUTPUT);

    OptimizerAttrs optimizer_attrs = OptimizerAttrs{
        SGDOptimizerAttrs{
            /*lr=*/0.0,
            /*momentum=*/0.0,
            /*nesterov=*/false,
            /*weight_decay=*/0.0,
        },
    };

    device_id_t device_idx =
        make_device_id_t_from_idx(nonnegative_int{0}, DeviceType::GPU);
    device_handle_t ff_handle =
        gpu_make_device_handle_t(managed_handle.raw_handle());

    std::unordered_map<DynamicValueAttrs, DynamicTensorAccessor> input_tensors;

    auto compute_loss = [&](LossAttrs const &loss_attrs) {
      ComputationGraphInstance computation_graph_instance =
          create_computation_graph_instance(
              /*cg=*/computation_graph,
              /*optimizer=*/optimizer_attrs,
              /*loss=*/loss_attrs,
              /*label_tensor=*/label_tensor,
              /*logit_tensor=*/dynamic_tensor_guid_t{logit_tensor},
              /*input_tensors=*/input_tensors,
              /*allocator=*/allocator,
              /*profiling_settings=*/ProfilingSettings{0, 1},
              /*device_handle=*/ff_handle,
              /*iteration_config=*/FFIterationConfig{1_p},
              /*device_idx=*/device_idx);

      perform_all_passes_for_computation_graph_instance(
          /*instance=*/computation_graph_instance,
          /*profiling_settings=*/ProfilingSettings{0, 0},
          /*ff_handle=*/ff_handle,
          /*iteration_config=*/FFIterationConfig{1_p},
          /*device_idx=*/device_idx);
      assert_unwrap(computation_graph_instance.get_loss_tensor_accessor());
    };

    SUBCASE("SparseCategoricalCrossEntropyLossAttrs") {
      TensorShape label_tensor_shape =
          TensorShape{TensorDims{FFOrdered{batch_size, 1_p}}, DataType::FLOAT};

      LossAttrs loss_attrs = LossAttrs{
          SparseCategoricalCrossEntropyLossAttrs{/*replace_labels=*/false}};

      compute_loss(loss_attrs);
    }

    SUBCASE("NonconfigurableLossAttrs") {
      TensorShape label_tensor_shape = TensorShape{
          TensorDims{FFOrdered{batch_size, output_dim}}, DataType::FLOAT};

      SUBCASE("LossFunction::CATEGORICAL_CROSSENTROPY") {
        LossAttrs loss_attrs = LossAttrs{
            NonconfigurableLossAttrs{LossFunction::CATEGORICAL_CROSSENTROPY}};

        compute_loss(loss_attrs);
      }

      SUBCASE("LossFunction::MEAN_SQUARED_ERROR_AVG_REDUCE") {
        LossAttrs loss_attrs = LossAttrs{NonconfigurableLossAttrs{
            LossFunction::MEAN_SQUARED_ERROR_AVG_REDUCE}};

        compute_loss(loss_attrs);
      }

      SUBCASE("LossFunction::IDENTITY") {
        LossAttrs loss_attrs =
            LossAttrs{NonconfigurableLossAttrs{LossFunction::IDENTITY}};

        compute_loss(loss_attrs);
      }
    }
  }
}
