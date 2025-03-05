#include "doctest/doctest.h"
#include "kernels/local_cuda_allocator.h"
#include "kernels/managed_ff_stream.h"
#include "kernels/managed_per_device_ff_handle.h"
#include "local-execution/allocated_tensors.h"
#include "local-execution/local_training_backing.h"
#include "pcg/computation_graph.h"
#include "pcg/computation_graph_builder.h"
#include "pcg/optimizer_attrs.dtg.h"
#include "test_utils.h"

namespace FlexFlow {

TEST_SUITE(FF_CUDA_TEST_SUITE) {
  TEST_CASE("Execute Update") {
    Allocator allocator = create_local_cuda_memory_allocator();
    AllocatedTensors allocated_tensors = make_empty_allocated_tensors();

    // construct computation graph
    ComputationGraph computation_graph = make_empty_computation_graph();

    nonnegative_int batch_size = 10_n;
    nonnegative_int data_dim = 16_n;
    nonnegative_int output_dim = 32_n;

    TensorShape input_tensor_shape = TensorShape{
        TensorDims{FFOrdered<nonnegative_int>{batch_size, data_dim}},
        DataType::FLOAT};

    TensorShape weight_shape = TensorShape{
        TensorDims{FFOrdered<nonnegative_int>{data_dim, output_dim}},
        DataType::FLOAT};

    LayerAddedResult inputs_layer = add_layer(
        computation_graph,
        LayerAttrs{ComputationGraphOpAttrs{InputAttrs{input_tensor_shape}},
                   "inputs"},
        {},
        {});

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
                                                       /*use_bias=*/true,
                                                       DataType::FLOAT,
                                                       std::nullopt,
                                                       std::nullopt}},
                   "linear"},
        inputs_layer.outputs,
        {});

    // initialize runtime configs
    ManagedPerDeviceFFHandle managed_handle{};

    RuntimeArgConfig runtime_arg_config = RuntimeArgConfig{
        DeviceSpecific<PerDeviceFFHandle>::create(managed_handle.raw_handle()),
        EnableProfiling::YES,
        ProfilingSettings{/*warmup_iters=*/0, /*measure_iters=*/1}};

    SUBCASE("SGDOptimizerAttrs") {
      SUBCASE("momentum=0") {
        OptimizerAttrs optimizer_attrs =
            OptimizerAttrs{SGDOptimizerAttrs{/*lr=*/0.001,
                                             /*momentum=*/0.0f,
                                             /*nesterov=*/false,
                                             /*weight_decay=*/0.001}};
        LocalTrainingBacking local_training_backing =
            LocalTrainingBacking{allocator,
                                 allocated_tensors,
                                 computation_graph,
                                 runtime_arg_config,
                                 optimizer_attrs};
        execute_update(local_training_backing,
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
        LocalTrainingBacking local_training_backing =
            LocalTrainingBacking{allocator,
                                 allocated_tensors,
                                 computation_graph,
                                 runtime_arg_config,
                                 optimizer_attrs};
        execute_update(local_training_backing,
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
      LocalTrainingBacking local_training_backing =
          LocalTrainingBacking{allocator,
                               allocated_tensors,
                               computation_graph,
                               runtime_arg_config,
                               optimizer_attrs};
      execute_update(local_training_backing,
                     linear_operator.layer,
                     optimizer_attrs,
                     allocator);
    }
  }
}

} // namespace FlexFlow
