#include "doctest/doctest.h"
#include "kernels/local_cuda_allocator.h"
#include "kernels/managed_per_device_ff_handle.h"
#include "local-execution/local_cost_estimator.h"
#include "op-attrs/ops/attention.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "pcg/computation_graph_builder.h"
#include "internal/test_utils.h"
#include "pcg/machine_view.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_CUDA_TEST_SUITE) {
  TEST_CASE("LocalCostEstimator") {
    ManagedPerDeviceFFHandle managed_handle = initialize_single_gpu_handle(
        /*workSpaceSize=*/1024 * 1024,
        /*allowTensorOpMathConversion=*/true);

    RuntimeArgConfig runtime_arg_config = RuntimeArgConfig{
        DeviceSpecific<PerDeviceFFHandle>::create(managed_handle.raw_handle()),
        EnableProfiling::YES,
        ProfilingSettings{/*warmup_iters=*/0,
                          /*measure_iters=*/1},
        DeviceType::GPU};

    OptimizerAttrs optimizer_attrs = OptimizerAttrs{
      SGDOptimizerAttrs{
        /*lr=*/0.1,
        /*momentum=*/0.1,
        /*nesterov=*/false,
        /*weight_decay=*/0.1,
      },
    };


    CostEstimator cost_estimator = get_local_cost_estimator(runtime_arg_config, optimizer_attrs);

    SUBCASE("estimate operator cost") {
      positive_int embed_dim = 32_p;
      positive_int num_heads = 10_p;
      MultiHeadAttentionAttrs attrs = MultiHeadAttentionAttrs{
          /*embed_dim=*/embed_dim,
          /*num_heads=*/num_heads,
          /*kdim=*/embed_dim,
          /*vdim=*/embed_dim,
          /*dropout=*/0.0,
          /*bias=*/false,
          /*add_bias_kv=*/false,
          /*add_zero_attn=*/false,
      };

      positive_int batch_size = 40_p;
      positive_int seq_len = 48_p;
      positive_int feature_size = 36_p;

      DataType dtype = DataType::FLOAT;
      ParallelTensorShape inputs_shape = lift_to_parallel(TensorShape{
          TensorDims{
              FFOrdered<positive_int>{batch_size, seq_len, feature_size}},
          DataType::FLOAT,
      });

      ParallelTensorShape weights_shape = throw_if_unexpected(
          get_weights_shape(attrs, inputs_shape, inputs_shape, inputs_shape));

      ParallelTensorShape output_shape = throw_if_unexpected(
          get_output_shape(attrs, inputs_shape, inputs_shape, inputs_shape));

      OpCostEstimateKey op_cost_estimate_key = OpCostEstimateKey{
          /*op_attrs=*/PCGOperatorAttrs{attrs},
          /*input_shapes=*/{inputs_shape, inputs_shape, inputs_shape},
          /*weight_shapes=*/{weights_shape},
          /*output_shapes=*/{output_shape},
          /*machine_view=*/make_1d_machine_view(
              MachineSpaceCoordinate{0_n, 0_n, DeviceType::GPU},
              MachineSpecificationDimension::INTRA_NODE,
              stride_t{1_p}),
      };

      OpCostMetrics result = cost_estimator.estimate_cost(op_cost_estimate_key);

      CHECK(result.forward_runtime > 0_ms);
      CHECK(result.backward_runtime > 0_ms);
      CHECK(result.memory_usage > 0_bytes);
    }
  }
}
