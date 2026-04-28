#include "internal/realm_test_utils.h"
#include "kernels/allocation.h"
#include "kernels/compare_tensor_accessors.h"
#include "kernels/copy_tensor_accessor.h"
#include "kernels/format_accessor_contents.h"
#include "kernels/tensor_accessor_reductions.h"
#include "op-attrs/operator_task_space_to_operator_task_space_mapping.h"
#include "op-attrs/ops/element_unary.h"
#include "op-attrs/ops/linear.h"
#include "op-attrs/ops/reduction.h"
#include "op-attrs/ops/repartition.h"
#include "op-attrs/ops/replicate.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "op-attrs/tensor_shape.dtg.h"
#include "op-attrs/tensor_slot_name.dtg.h"
#include "pcg/device_type.dtg.h"
#include "pcg/machine_space_coordinate.dtg.h"
#include "pcg/mapped_parallel_computation_graph/operator_atomic_task_shard_binding.dtg.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph_builder.h"
#include "pcg/parallel_computation_graph/parallel_layer_guid_t.dtg.h"
#include "pcg/parallel_computation_graph/parallel_tensor_guid_t.dtg.h"
#include "realm-execution/distributed_ff_handle.h"
#include "realm-execution/dynamic_tensor_accessor_from_instance.h"
#include "realm-execution/pcg_instance.h"
#include "realm-execution/realm_context.h"
#include "realm-execution/realm_manager.h"
#include "task-spec/permissions.h"
#include "test/utils/doctest/check_kv.h"
#include "utils/containers/require_only_key.h"
#include <doctest/doctest.h>

namespace test {

using namespace ::FlexFlow;
namespace Realm = ::FlexFlow::Realm;

template <typename T>
static ParallelLayerAttrs make_layer_attrs(T const &op_attrs) {
  return ParallelLayerAttrs{
      /*op_attrs=*/PCGOperatorAttrs{op_attrs},
      /*name=*/std::nullopt,
  };
};

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("RealmBackend e2e Training Reduction Op (CPU Model Parallelism)") {
    std::vector<char *> fake_args =
        make_fake_realm_args(/*num_cpus=*/2_p, /*num_gpus=*/0_n);
    int fake_argc = fake_args.size();
    char **fake_argv = fake_args.data();

    RealmManager manager = RealmManager{&fake_argc, &fake_argv};
    ControllerTaskResult result = manager.start_controller([](RealmContext
                                                                  &ctx) {
      Allocator allocator = ctx.get_current_device_allocator();

      positive_int batch_size = 4_p;
      positive_int in_channels = 8_p;
      positive_int out_channels = 4_p;

      TensorShape input_tensor_shape = TensorShape{
          TensorDims{FFOrdered{batch_size, in_channels}}, DataType::FLOAT};

      TensorShape weight_tensor_shape = TensorShape{
          TensorDims{FFOrdered{out_channels, in_channels}}, DataType::FLOAT};

      ParallelComputationGraph pcg = empty_parallel_computation_graph();

      // input layer
      ParallelLayerAddedResult inputs_layer =
          pcg_add_input_layer(pcg, input_tensor_shape);
      parallel_tensor_guid_t t_input =
          require_only_key(inputs_layer.outputs, TensorSlotName::OUTPUT);

      // weight layer
      ParallelLayerAddedResult weights_layer =
          pcg_add_input_layer(pcg, weight_tensor_shape);
      parallel_tensor_guid_t t_weight =
          require_only_key(weights_layer.outputs, TensorSlotName::OUTPUT);

      // repartition input along feature dim (dim 1) with degree 2
      RepartitionAttrs input_repartition_attrs{
          /*repartition_dim=*/ff_dim_t{nonnegative_int{1}},
          /*repartition_degree=*/2_p,
      };
      ParallelLayerAddedResult input_repartition_operator =
          add_parallel_layer(pcg,
                             make_layer_attrs(input_repartition_attrs),
                             {{TensorSlotName::INPUT, t_input}},
                             /*weights=*/{});
      parallel_tensor_guid_t t_input_repartitioned = require_only_key(
          input_repartition_operator.outputs, TensorSlotName::OUTPUT);

      // repartition weight along feature dim (dim 1) with degree 2
      // to match the repartitioned input
      RepartitionAttrs weight_repartition_attrs{
          /*repartition_dim=*/ff_dim_t{nonnegative_int{1}},
          /*repartition_degree=*/2_p,
      };
      ParallelLayerAddedResult weight_repartition_operator =
          add_parallel_layer(pcg,
                             make_layer_attrs(weight_repartition_attrs),
                             {{TensorSlotName::INPUT, t_weight}},
                             /*weights=*/{});
      parallel_tensor_guid_t t_weight_repartitioned = require_only_key(
          weight_repartition_operator.outputs, TensorSlotName::OUTPUT);

      // linear with repartitioned input and weight
      // shard_dim[-1]=2 → sum_degree=2 output
      ParallelLayerAddedResult linear_operator = add_parallel_layer(
          pcg,
          ParallelLayerAttrs{PCGOperatorAttrs{LinearAttrs{out_channels,
                                                          /*use_bias=*/false,
                                                          DataType::FLOAT,
                                                          Activation::RELU,
                                                          std::nullopt}},
                             std::nullopt},
          /*inputs=*/
          {
              {TensorSlotName::INPUT, t_input_repartitioned},
          },
          /*weights=*/
          {
              {TensorSlotName::WEIGHT, t_weight_repartitioned},
          });
      parallel_tensor_guid_t t_linear =
          require_only_key(linear_operator.outputs, TensorSlotName::OUTPUT);

      // reduction degree=2 — sums partial results
      ReductionAttrs reduction_attrs{/*reduction_degree=*/2_p};
      ParallelLayerAddedResult reduction_operator =
          add_parallel_layer(pcg,
                             make_layer_attrs(reduction_attrs),
                             {{TensorSlotName::INPUT, t_linear}},
                             /*weights=*/{});
      parallel_tensor_guid_t t_reduced =
          require_only_key(reduction_operator.outputs, TensorSlotName::OUTPUT);

      // relu consumer
      ParallelLayerAddedResult relu_operator =
          add_parallel_layer(pcg,
                             make_layer_attrs(make_relu_attrs()),
                             {{TensorSlotName::INPUT, t_reduced}},
                             /*weights=*/{});

      MachineSpaceCoordinate cpu0{0_n, 0_n, DeviceType::CPU};
      MachineSpaceCoordinate cpu1{0_n, 1_n, DeviceType::CPU};

      // input: unsharded on cpu0 — 2 shard dims
      ParallelTensorSpaceCoordinate input_coord{0_n, 0_n, FFOrdered{0_n, 0_n}};

      // weight: unsharded on cpu0 — 2 shard dims
      ParallelTensorSpaceCoordinate weight_coord{0_n, 0_n, FFOrdered{0_n, 0_n}};

      // after repartition: input sharded along feature dim
      ParallelTensorSpaceCoordinate input_repartitioned_coord_0{
          0_n, 0_n, FFOrdered{0_n, 0_n}};
      ParallelTensorSpaceCoordinate input_repartitioned_coord_1{
          0_n, 0_n, FFOrdered{0_n, 1_n}};

      // after repartition: weight sharded along feature dim
      ParallelTensorSpaceCoordinate weight_repartitioned_coord_0{
          0_n, 0_n, FFOrdered{0_n, 0_n}};
      ParallelTensorSpaceCoordinate weight_repartitioned_coord_1{
          0_n, 0_n, FFOrdered{0_n, 1_n}};

      // linear output: partial sums — sum_component distinguishes them
      // output has 2 shard dims [{4,1},{4,1}]
      ParallelTensorSpaceCoordinate linear_coord_0{
          0_n, 0_n, FFOrdered{0_n, 0_n}};
      ParallelTensorSpaceCoordinate linear_coord_1{
          1_n, 0_n, FFOrdered{0_n, 0_n}};

      // reduced output: fully reduced on cpu0
      ParallelTensorSpaceCoordinate reduced_coord{
          0_n, 0_n, FFOrdered{0_n, 0_n}};

      MappedParallelComputationGraph mpcg{
          pcg,
          {
              // input: unsharded on cpu0
              {inputs_layer.parallel_layer,
               MappedOperatorTaskGroup{
                   {{cpu0,
                     OperatorAtomicTaskShardBinding{
                         {{TensorSlotName::OUTPUT, input_coord}}}}}}},
              // weight: unsharded on cpu0
              {weights_layer.parallel_layer,
               MappedOperatorTaskGroup{
                   {{cpu0,
                     OperatorAtomicTaskShardBinding{
                         {{TensorSlotName::OUTPUT, weight_coord}}}}}}},
              // input repartition: OUTPUT only
              {input_repartition_operator.parallel_layer,
               MappedOperatorTaskGroup{{
                   {cpu0,
                    OperatorAtomicTaskShardBinding{{
                        {TensorSlotName::OUTPUT, input_repartitioned_coord_0},
                    }}},
                   {cpu1,
                    OperatorAtomicTaskShardBinding{{
                        {TensorSlotName::OUTPUT, input_repartitioned_coord_1},
                    }}},
               }}},
              // weight repartition: OUTPUT only
              {weight_repartition_operator.parallel_layer,
               MappedOperatorTaskGroup{{
                   {cpu0,
                    OperatorAtomicTaskShardBinding{{
                        {TensorSlotName::OUTPUT, weight_repartitioned_coord_0},
                    }}},
                   {cpu1,
                    OperatorAtomicTaskShardBinding{{
                        {TensorSlotName::OUTPUT, weight_repartitioned_coord_1},
                    }}},
               }}},
              // linear: INPUT + WEIGHT + OUTPUT per device
              {linear_operator.parallel_layer,
               MappedOperatorTaskGroup{{
                   {cpu0,
                    OperatorAtomicTaskShardBinding{{
                        {TensorSlotName::INPUT, input_repartitioned_coord_0},
                        {TensorSlotName::WEIGHT, weight_repartitioned_coord_0},
                        {TensorSlotName::OUTPUT, linear_coord_0},
                    }}},
                   {cpu1,
                    OperatorAtomicTaskShardBinding{{
                        {TensorSlotName::INPUT, input_repartitioned_coord_1},
                        {TensorSlotName::WEIGHT, weight_repartitioned_coord_1},
                        {TensorSlotName::OUTPUT, linear_coord_1},
                    }}},
               }}},
              // reduction: INPUT only — OUTPUT coords not distinct
              {reduction_operator.parallel_layer,
               MappedOperatorTaskGroup{{
                   {cpu0,
                    OperatorAtomicTaskShardBinding{{
                        {TensorSlotName::INPUT, linear_coord_0},
                    }}},
                   {cpu1,
                    OperatorAtomicTaskShardBinding{{
                        {TensorSlotName::INPUT, linear_coord_1},
                    }}},
               }}},
              // relu: on cpu0 only
              {relu_operator.parallel_layer,
               MappedOperatorTaskGroup{{
                   {cpu0,
                    OperatorAtomicTaskShardBinding{{
                        {TensorSlotName::INPUT, reduced_coord},
                        {TensorSlotName::OUTPUT, reduced_coord},
                    }}},
               }}},
          }};

      OptimizerAttrs optimizer_attrs =
          OptimizerAttrs{SGDOptimizerAttrs{/*lr=*/0.001,
                                           /*momentum=*/0.9,
                                           /*nesterov=*/false,
                                           /*weight_decay=*/0.001}};

      std::unordered_map<DynamicValueAttrs, DynamicTensorAccessor>
          input_tensors;

      DistributedFfHandle device_handle =
          create_distributed_ff_handle(ctx,
                                       /*workSpaceSize=*/1024 * 1024,
                                       /*allowTensorOpMathConversion=*/true);

      PCGInstance pcg_instance = create_pcg_instance(ctx,
                                                     mpcg,
                                                     optimizer_attrs,
                                                     std::nullopt,
                                                     input_tensors,
                                                     ProfilingSettings{0, 0},
                                                     device_handle,
                                                     FFIterationConfig{1_p});

      perform_all_passes_for_pcg_instance(pcg_instance,
                                          ProfilingSettings{0, 0},
                                          device_handle,
                                          FFIterationConfig{1_p});
    });
    result.wait();
  }
}
TEST_SUITE(FF_CUDA_TEST_SUITE) {
  TEST_CASE("RealmBackend e2e Training Reduction Op (GPU Model Parallelism)") {
    std::vector<char *> fake_args =
        make_fake_realm_args(/*num_cpus=*/1_p, /*num_gpus=*/2_n);
    int fake_argc = fake_args.size();
    char **fake_argv = fake_args.data();

    RealmManager manager = RealmManager{&fake_argc, &fake_argv};
    ControllerTaskResult result = manager.start_controller([](RealmContext
                                                                  &ctx) {
      Allocator allocator = ctx.get_current_device_allocator();

      positive_int batch_size = 4_p;
      positive_int in_channels = 8_p;
      positive_int out_channels = 4_p;

      TensorShape input_tensor_shape = TensorShape{
          TensorDims{FFOrdered{batch_size, in_channels}}, DataType::FLOAT};

      TensorShape weight_tensor_shape = TensorShape{
          TensorDims{FFOrdered{out_channels, in_channels}}, DataType::FLOAT};

      ParallelComputationGraph pcg = empty_parallel_computation_graph();

      // input layer
      ParallelLayerAddedResult inputs_layer =
          pcg_add_input_layer(pcg, input_tensor_shape);
      parallel_tensor_guid_t t_input =
          require_only_key(inputs_layer.outputs, TensorSlotName::OUTPUT);

      // weight layer
      ParallelLayerAddedResult weights_layer =
          pcg_add_input_layer(pcg, weight_tensor_shape);
      parallel_tensor_guid_t t_weight =
          require_only_key(weights_layer.outputs, TensorSlotName::OUTPUT);

      // repartition input along feature dim (dim 1) with degree 2
      RepartitionAttrs input_repartition_attrs{
          /*repartition_dim=*/ff_dim_t{nonnegative_int{1}},
          /*repartition_degree=*/2_p,
      };
      ParallelLayerAddedResult input_repartition_operator =
          add_parallel_layer(pcg,
                             make_layer_attrs(input_repartition_attrs),
                             {{TensorSlotName::INPUT, t_input}},
                             /*weights=*/{});
      parallel_tensor_guid_t t_input_repartitioned = require_only_key(
          input_repartition_operator.outputs, TensorSlotName::OUTPUT);

      // repartition weight along feature dim (dim 1) with degree 2
      // to match the repartitioned input
      RepartitionAttrs weight_repartition_attrs{
          /*repartition_dim=*/ff_dim_t{nonnegative_int{1}},
          /*repartition_degree=*/2_p,
      };
      ParallelLayerAddedResult weight_repartition_operator =
          add_parallel_layer(pcg,
                             make_layer_attrs(weight_repartition_attrs),
                             {{TensorSlotName::INPUT, t_weight}},
                             /*weights=*/{});
      parallel_tensor_guid_t t_weight_repartitioned = require_only_key(
          weight_repartition_operator.outputs, TensorSlotName::OUTPUT);

      // linear with repartitioned input and weight
      // shard_dim[-1]=2 → sum_degree=2 output
      ParallelLayerAddedResult linear_operator = add_parallel_layer(
          pcg,
          ParallelLayerAttrs{PCGOperatorAttrs{LinearAttrs{out_channels,
                                                          /*use_bias=*/false,
                                                          DataType::FLOAT,
                                                          Activation::RELU,
                                                          std::nullopt}},
                             std::nullopt},
          /*inputs=*/
          {
              {TensorSlotName::INPUT, t_input_repartitioned},
          },
          /*weights=*/
          {
              {TensorSlotName::WEIGHT, t_weight_repartitioned},
          });
      parallel_tensor_guid_t t_linear =
          require_only_key(linear_operator.outputs, TensorSlotName::OUTPUT);

      // reduction degree=2 — sums partial results
      ReductionAttrs reduction_attrs{/*reduction_degree=*/2_p};
      ParallelLayerAddedResult reduction_operator =
          add_parallel_layer(pcg,
                             make_layer_attrs(reduction_attrs),
                             {{TensorSlotName::INPUT, t_linear}},
                             /*weights=*/{});
      parallel_tensor_guid_t t_reduced =
          require_only_key(reduction_operator.outputs, TensorSlotName::OUTPUT);

      // relu consumer
      ParallelLayerAddedResult relu_operator =
          add_parallel_layer(pcg,
                             make_layer_attrs(make_relu_attrs()),
                             {{TensorSlotName::INPUT, t_reduced}},
                             /*weights=*/{});

      MachineSpaceCoordinate gpu0{0_n, 0_n, DeviceType::GPU};
      MachineSpaceCoordinate gpu1{0_n, 1_n, DeviceType::GPU};

      // input: unsharded on gpu0 — 2 shard dims
      ParallelTensorSpaceCoordinate input_coord{0_n, 0_n, FFOrdered{0_n, 0_n}};

      // weight: unsharded on gpu0 — 2 shard dims
      ParallelTensorSpaceCoordinate weight_coord{0_n, 0_n, FFOrdered{0_n, 0_n}};

      // after repartition: input sharded along feature dim
      ParallelTensorSpaceCoordinate input_repartitioned_coord_0{
          0_n, 0_n, FFOrdered{0_n, 0_n}};
      ParallelTensorSpaceCoordinate input_repartitioned_coord_1{
          0_n, 0_n, FFOrdered{0_n, 1_n}};

      // after repartition: weight sharded along feature dim
      ParallelTensorSpaceCoordinate weight_repartitioned_coord_0{
          0_n, 0_n, FFOrdered{0_n, 0_n}};
      ParallelTensorSpaceCoordinate weight_repartitioned_coord_1{
          0_n, 0_n, FFOrdered{0_n, 1_n}};

      // linear output: partial sums — sum_component distinguishes them
      // output has 2 shard dims [{4,1},{4,1}]
      ParallelTensorSpaceCoordinate linear_coord_0{
          0_n, 0_n, FFOrdered{0_n, 0_n}};
      ParallelTensorSpaceCoordinate linear_coord_1{
          1_n, 0_n, FFOrdered{0_n, 0_n}};

      // reduced output: fully reduced on gpu0
      ParallelTensorSpaceCoordinate reduced_coord{
          0_n, 0_n, FFOrdered{0_n, 0_n}};

      MappedParallelComputationGraph mpcg{
          pcg,
          {
              // input: unsharded on gpu0
              {inputs_layer.parallel_layer,
               MappedOperatorTaskGroup{
                   {{gpu0,
                     OperatorAtomicTaskShardBinding{
                         {{TensorSlotName::OUTPUT, input_coord}}}}}}},
              // weight: unsharded on gpu0
              {weights_layer.parallel_layer,
               MappedOperatorTaskGroup{
                   {{gpu0,
                     OperatorAtomicTaskShardBinding{
                         {{TensorSlotName::OUTPUT, weight_coord}}}}}}},
              // input repartition: OUTPUT only
              {input_repartition_operator.parallel_layer,
               MappedOperatorTaskGroup{{
                   {gpu0,
                    OperatorAtomicTaskShardBinding{{
                        {TensorSlotName::OUTPUT, input_repartitioned_coord_0},
                    }}},
                   {gpu1,
                    OperatorAtomicTaskShardBinding{{
                        {TensorSlotName::OUTPUT, input_repartitioned_coord_1},
                    }}},
               }}},
              // weight repartition: OUTPUT only
              {weight_repartition_operator.parallel_layer,
               MappedOperatorTaskGroup{{
                   {gpu0,
                    OperatorAtomicTaskShardBinding{{
                        {TensorSlotName::OUTPUT, weight_repartitioned_coord_0},
                    }}},
                   {gpu1,
                    OperatorAtomicTaskShardBinding{{
                        {TensorSlotName::OUTPUT, weight_repartitioned_coord_1},
                    }}},
               }}},
              // linear: INPUT + WEIGHT + OUTPUT per device
              {linear_operator.parallel_layer,
               MappedOperatorTaskGroup{{
                   {gpu0,
                    OperatorAtomicTaskShardBinding{{
                        {TensorSlotName::INPUT, input_repartitioned_coord_0},
                        {TensorSlotName::WEIGHT, weight_repartitioned_coord_0},
                        {TensorSlotName::OUTPUT, linear_coord_0},
                    }}},
                   {gpu1,
                    OperatorAtomicTaskShardBinding{{
                        {TensorSlotName::INPUT, input_repartitioned_coord_1},
                        {TensorSlotName::WEIGHT, weight_repartitioned_coord_1},
                        {TensorSlotName::OUTPUT, linear_coord_1},
                    }}},
               }}},
              // reduction: INPUT only — OUTPUT coords not distinct
              {reduction_operator.parallel_layer,
               MappedOperatorTaskGroup{{
                   {gpu0,
                    OperatorAtomicTaskShardBinding{{
                        {TensorSlotName::INPUT, linear_coord_0},
                    }}},
                   {gpu1,
                    OperatorAtomicTaskShardBinding{{
                        {TensorSlotName::INPUT, linear_coord_1},
                    }}},
               }}},
              // relu: on gpu0 only
              {relu_operator.parallel_layer,
               MappedOperatorTaskGroup{{
                   {gpu0,
                    OperatorAtomicTaskShardBinding{{
                        {TensorSlotName::INPUT, reduced_coord},
                        {TensorSlotName::OUTPUT, reduced_coord},
                    }}},
               }}},
          }};

      OptimizerAttrs optimizer_attrs =
          OptimizerAttrs{SGDOptimizerAttrs{/*lr=*/0.001,
                                           /*momentum=*/0.9,
                                           /*nesterov=*/false,
                                           /*weight_decay=*/0.001}};

      std::unordered_map<DynamicValueAttrs, DynamicTensorAccessor>
          input_tensors;

      DistributedFfHandle device_handle =
          create_distributed_ff_handle(ctx,
                                       /*workSpaceSize=*/1024 * 1024,
                                       /*allowTensorOpMathConversion=*/true);

      PCGInstance pcg_instance = create_pcg_instance(ctx,
                                                     mpcg,
                                                     optimizer_attrs,
                                                     std::nullopt,
                                                     input_tensors,
                                                     ProfilingSettings{0, 0},
                                                     device_handle,
                                                     FFIterationConfig{1_p});

      perform_all_passes_for_pcg_instance(pcg_instance,
                                          ProfilingSettings{0, 0},
                                          device_handle,
                                          FFIterationConfig{1_p});
    });
    result.wait();
  }
}
} // namespace test
