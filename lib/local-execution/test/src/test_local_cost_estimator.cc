#include "doctest/doctest.h"
#include "kernels/local_cuda_allocator.h"
#include "kernels/managed_per_device_ff_handle.h"
#include "local-execution/local_cost_estimator.h"
#include "op-attrs/ops/attention.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "pcg/computation_graph_builder.h"
#include "test_utils.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_CUDA_TEST_SUITE) {
  TEST_CASE("Local Cost Estimator") {
    // local backing initialization
    ManagedPerDeviceFFHandle managed_handle{};

    RuntimeArgConfig runtime_arg_config = RuntimeArgConfig{
        DeviceSpecific<PerDeviceFFHandle>::create(managed_handle.raw_handle()),
        EnableProfiling::YES,
        ProfilingSettings{/*warmup_iters=*/0,
                          /*measure_iters=*/1}};

    LocalCostEstimator cost_estimator = LocalCostEstimator{runtime_arg_config};

    SUBCASE("Estimate cost -- Attention Op") {
      nonnegative_int embed_dim = 32_n;
      nonnegative_int num_heads = 10_n;
      MultiHeadAttentionAttrs attrs = MultiHeadAttentionAttrs{
          /*embed_dim=*/embed_dim,
          /*num_heads=*/num_heads,
          /*kdim=*/embed_dim,
          /*vdim=*/embed_dim,
          /*dropout=*/0.0,
          /*bias=*/true,
          /*add_bias_kv=*/false,
          /*add_zero_attn=*/false,
      };

      nonnegative_int batch_size = 40_n;
      nonnegative_int seq_len = 48_n;
      nonnegative_int feature_size = 36_n;

      DataType dtype = DataType::FLOAT;
      ParallelTensorShape inputs_shape = lift_to_parallel(TensorShape{
          TensorDims{
              FFOrdered<nonnegative_int>{batch_size, seq_len, feature_size}},
          DataType::FLOAT,
      });

      ParallelTensorShape weights_shape = throw_if_unexpected(
          get_weights_shape(attrs, inputs_shape, inputs_shape, inputs_shape));
      ParallelTensorAttrs weight_attrs =
          ParallelTensorAttrs{weights_shape, CreateGrad::YES};

      ParallelTensorShape output_shape = throw_if_unexpected(
          get_output_shape(attrs, inputs_shape, inputs_shape, inputs_shape));
      ParallelTensorAttrs output_attrs =
          ParallelTensorAttrs{output_shape, CreateGrad::YES};

      CostDetails result = cost_estimator.estimate_cost(
          PCGOperatorAttrs{attrs},
          std::vector<ParallelTensorShape>{
              inputs_shape, inputs_shape, inputs_shape},
          std::vector<ParallelTensorAttrs>{weight_attrs},
          std::vector<ParallelTensorAttrs>{output_attrs},
          make_1d_machine_view(
              MachineSpaceCoordinate{0_n, 0_n, DeviceType::GPU},
              MachineSpecificationDimension::INTRA_NODE,
              stride_t{0_n}));

      CHECK(result.total_elapsed_time > 0);
      CHECK(result.total_mem_usage > 0);
    }
  }
}
