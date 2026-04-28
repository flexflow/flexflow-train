#include "internal/realm_test_utils.h"
#include "kernels/allocation.h"
#include "kernels/compare_tensor_accessors.h"
#include "kernels/copy_tensor_accessor.h"
#include "kernels/format_accessor_contents.h"
#include "kernels/tensor_accessor_reductions.h"
#include "op-attrs/operator_task_space_to_operator_task_space_mapping.h"
#include "op-attrs/ops/combine.h"
#include "op-attrs/ops/element_unary.h"
#include "op-attrs/ops/linear.h"
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
  TEST_CASE("RealmBackend e2e Training Combine Op (CPU Model Parallelism)") {
    std::vector<char *> fake_args =
        make_fake_realm_args(/*num_cpus=*/2_p, /*num_gpus=*/0_n);
    int fake_argc = fake_args.size();
    char **fake_argv = fake_args.data();

    RealmManager manager = RealmManager{&fake_argc, &fake_argv};
    ControllerTaskResult result =
        manager.start_controller([](RealmContext &ctx) {
          Allocator allocator = ctx.get_current_device_allocator();

          positive_int batch_size = 10_p;
          positive_int data_dim = 16_p;

          TensorShape input_tensor_shape = TensorShape{
              TensorDims{FFOrdered{batch_size, data_dim}}, DataType::FLOAT};

          ParallelComputationGraph pcg = empty_parallel_computation_graph();

          // input layer
          ParallelLayerAddedResult inputs_layer =
              pcg_add_input_layer(pcg, input_tensor_shape);
          parallel_tensor_guid_t t_input =
              require_only_key(inputs_layer.outputs, TensorSlotName::OUTPUT);

          // repartition along dim 0 with degree 2
          // needed so combine has a degree=2 sharded tensor to combine
          RepartitionAttrs repartition_attrs{
              /*repartition_dim=*/ff_dim_t{nonnegative_int{0}},
              /*repartition_degree=*/2_p,
          };
          ParallelLayerAddedResult repartition_operator =
              add_parallel_layer(pcg,
                                 make_layer_attrs(repartition_attrs),
                                 {{TensorSlotName::INPUT, t_input}},
                                 /*weights=*/{});
          parallel_tensor_guid_t t_repartitioned = require_only_key(
              repartition_operator.outputs, TensorSlotName::OUTPUT);

          // combine along dim 0 with degree 2
          CombineAttrs combine_attrs{
              /*combine_dim=*/ff_dim_t{nonnegative_int{0}},
              /*combine_degree=*/2_p,
          };
          ParallelLayerAddedResult combine_operator =
              add_parallel_layer(pcg,
                                 make_layer_attrs(combine_attrs),
                                 {{TensorSlotName::INPUT, t_repartitioned}},
                                 /*weights=*/{});
          parallel_tensor_guid_t t_combined = require_only_key(
              combine_operator.outputs, TensorSlotName::OUTPUT);

          // relu consumer
          ParallelLayerAddedResult relu_operator =
              add_parallel_layer(pcg,
                                 make_layer_attrs(make_relu_attrs()),
                                 {{TensorSlotName::INPUT, t_combined}},
                                 /*weights=*/{});

          MachineSpaceCoordinate cpu0{0_n, 0_n, DeviceType::CPU};
          MachineSpaceCoordinate cpu1{0_n, 1_n, DeviceType::CPU};

          // input: one shard on cpu0 (not yet repartitioned)
          ParallelTensorSpaceCoordinate tensor_coord0{
              0_n, 0_n, FFOrdered{0_n, 0_n}};
          // after repartition: two shards along dim 0
          ParallelTensorSpaceCoordinate tensor_coord_shard0{
              0_n, 0_n, FFOrdered{0_n, 0_n}};
          ParallelTensorSpaceCoordinate tensor_coord_shard1{
              0_n, 0_n, FFOrdered{1_n, 0_n}};
          // after combine: one shard on cpu0
          ParallelTensorSpaceCoordinate tensor_coord_combined{
              0_n, 0_n, FFOrdered{0_n, 0_n}};

          MappedParallelComputationGraph mpcg{
              pcg,
              {
                  // input: one shard on cpu0
                  {inputs_layer.parallel_layer,
                   MappedOperatorTaskGroup{
                       {{cpu0,
                         OperatorAtomicTaskShardBinding{
                             {{TensorSlotName::OUTPUT, tensor_coord0}}}}}}},
                  // repartition: OUTPUT only — no INPUT since all replicas
                  // read same source coord violating bidict uniqueness
                  {repartition_operator.parallel_layer,
                   MappedOperatorTaskGroup{{
                       {cpu0,
                        OperatorAtomicTaskShardBinding{{
                            {TensorSlotName::OUTPUT, tensor_coord_shard0},
                        }}},
                       {cpu1,
                        OperatorAtomicTaskShardBinding{{
                            {TensorSlotName::OUTPUT, tensor_coord_shard1},
                        }}},
                   }}},
                  // combine: two inputs → one output on cpu0
                  {combine_operator.parallel_layer,
                   MappedOperatorTaskGroup{{
                       {cpu0,
                        OperatorAtomicTaskShardBinding{{
                            {TensorSlotName::INPUT, tensor_coord_shard0},
                        }}},
                       {cpu1,
                        OperatorAtomicTaskShardBinding{{
                            {TensorSlotName::INPUT, tensor_coord_shard1},
                        }}},
                   }}},
                  // relu: one shard on cpu0
                  {relu_operator.parallel_layer,
                   MappedOperatorTaskGroup{{
                       {cpu0,
                        OperatorAtomicTaskShardBinding{{
                            {TensorSlotName::INPUT, tensor_coord_combined},
                            {TensorSlotName::OUTPUT, tensor_coord_combined},
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

          DistributedFfHandle device_handle = create_distributed_ff_handle(
              ctx,
              /*workSpaceSize=*/1024 * 1024,
              /*allowTensorOpMathConversion=*/true);

          PCGInstance pcg_instance =
              create_pcg_instance(ctx,
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
  TEST_CASE("RealmBackend e2e Training Combine Op (GPU Model Parallelism)") {
    std::vector<char *> fake_args =
        make_fake_realm_args(/*num_cpus=*/1_p, /*num_gpus=*/2_n);
    int fake_argc = fake_args.size();
    char **fake_argv = fake_args.data();

    RealmManager manager = RealmManager{&fake_argc, &fake_argv};

    ControllerTaskResult result =
        manager.start_controller([](RealmContext &ctx) {
          Allocator allocator = ctx.get_current_device_allocator();

          positive_int batch_size = 10_p;
          positive_int data_dim = 16_p;

          TensorShape input_tensor_shape = TensorShape{
              TensorDims{FFOrdered{batch_size, data_dim}}, DataType::FLOAT};

          ParallelComputationGraph pcg = empty_parallel_computation_graph();

          // input layer
          ParallelLayerAddedResult inputs_layer =
              pcg_add_input_layer(pcg, input_tensor_shape);
          parallel_tensor_guid_t t_input =
              require_only_key(inputs_layer.outputs, TensorSlotName::OUTPUT);

          // repartition along dim 0 with degree 2
          // needed so combine has a degree=2 sharded tensor to combine
          RepartitionAttrs repartition_attrs{
              /*repartition_dim=*/ff_dim_t{nonnegative_int{0}},
              /*repartition_degree=*/2_p,
          };
          ParallelLayerAddedResult repartition_operator =
              add_parallel_layer(pcg,
                                 make_layer_attrs(repartition_attrs),
                                 {{TensorSlotName::INPUT, t_input}},
                                 /*weights=*/{});
          parallel_tensor_guid_t t_repartitioned = require_only_key(
              repartition_operator.outputs, TensorSlotName::OUTPUT);

          // combine along dim 0 with degree 2
          CombineAttrs combine_attrs{
              /*combine_dim=*/ff_dim_t{nonnegative_int{0}},
              /*combine_degree=*/2_p,
          };
          ParallelLayerAddedResult combine_operator =
              add_parallel_layer(pcg,
                                 make_layer_attrs(combine_attrs),
                                 {{TensorSlotName::INPUT, t_repartitioned}},
                                 /*weights=*/{});
          parallel_tensor_guid_t t_combined = require_only_key(
              combine_operator.outputs, TensorSlotName::OUTPUT);

          // relu consumer
          ParallelLayerAddedResult relu_operator =
              add_parallel_layer(pcg,
                                 make_layer_attrs(make_relu_attrs()),
                                 {{TensorSlotName::INPUT, t_combined}},
                                 /*weights=*/{});

          MachineSpaceCoordinate gpu0{0_n, 0_n, DeviceType::GPU};
          MachineSpaceCoordinate gpu1{0_n, 1_n, DeviceType::GPU};

          // input: one shard on gpu0 (not yet repartitioned)
          ParallelTensorSpaceCoordinate tensor_coord0{
              0_n, 0_n, FFOrdered{0_n, 0_n}};
          // after repartition: two shards along dim 0
          ParallelTensorSpaceCoordinate tensor_coord_shard0{
              0_n, 0_n, FFOrdered{0_n, 0_n}};
          ParallelTensorSpaceCoordinate tensor_coord_shard1{
              0_n, 0_n, FFOrdered{1_n, 0_n}};
          // after combine: one shard on gpu0
          ParallelTensorSpaceCoordinate tensor_coord_combined{
              0_n, 0_n, FFOrdered{0_n, 0_n}};

          MappedParallelComputationGraph mpcg{
              pcg,
              {
                  // input: one shard on gpu0
                  {inputs_layer.parallel_layer,
                   MappedOperatorTaskGroup{
                       {{gpu0,
                         OperatorAtomicTaskShardBinding{
                             {{TensorSlotName::OUTPUT, tensor_coord0}}}}}}},
                  // repartition: OUTPUT only — no INPUT since all replicas
                  // read same source coord violating bidict uniqueness
                  {repartition_operator.parallel_layer,
                   MappedOperatorTaskGroup{{
                       {gpu0,
                        OperatorAtomicTaskShardBinding{{
                            {TensorSlotName::OUTPUT, tensor_coord_shard0},
                        }}},
                       {gpu1,
                        OperatorAtomicTaskShardBinding{{
                            {TensorSlotName::OUTPUT, tensor_coord_shard1},
                        }}},
                   }}},
                  // combine: two inputs → one output on gpu0
                  {combine_operator.parallel_layer,
                   MappedOperatorTaskGroup{{
                       {gpu0,
                        OperatorAtomicTaskShardBinding{{
                            {TensorSlotName::INPUT, tensor_coord_shard0},
                        }}},
                       {gpu1,
                        OperatorAtomicTaskShardBinding{{
                            {TensorSlotName::INPUT, tensor_coord_shard1},
                        }}},
                   }}},
                  // relu: one shard on gpu0
                  {relu_operator.parallel_layer,
                   MappedOperatorTaskGroup{{
                       {gpu0,
                        OperatorAtomicTaskShardBinding{{
                            {TensorSlotName::INPUT, tensor_coord_combined},
                            {TensorSlotName::OUTPUT, tensor_coord_combined},
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

          DistributedFfHandle device_handle = create_distributed_ff_handle(
              ctx,
              /*workSpaceSize=*/1024 * 1024,
              /*allowTensorOpMathConversion=*/true);

          PCGInstance pcg_instance =
              create_pcg_instance(ctx,
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
