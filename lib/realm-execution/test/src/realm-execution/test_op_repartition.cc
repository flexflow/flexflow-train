#include "internal/realm_test_utils.h"
#include "kernels/allocation.h"
#include "kernels/compare_tensor_accessors.h"
#include "kernels/copy_tensor_accessor.h"
#include "kernels/format_accessor_contents.h"
#include "kernels/tensor_accessor_reductions.h"
#include "op-attrs/operator_task_space_to_operator_task_space_mapping.h"
#include "op-attrs/ops/element_unary.h"
#include "op-attrs/ops/linear.h"
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
  TEST_CASE("RealmBackend Repartition Op with External Input Instance (CPU)") {
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
          int num_elements = batch_size.int_from_positive_int() *
                             data_dim.int_from_positive_int();

          TensorShape input_tensor_shape = TensorShape{
              TensorDims{FFOrdered{batch_size, data_dim}}, DataType::FLOAT};

          // allocate external input and fill with known values
          GenericTensorAccessorW input_tensor =
              allocator.allocate_tensor(input_tensor_shape);
          float *input_ptr = input_tensor.get_float_ptr();
          for (int i = 0; i < num_elements; i++) {
            input_ptr[i] = static_cast<float>(i);
          }

          // same PCG as existing test
          ParallelComputationGraph pcg = empty_parallel_computation_graph();

          ParallelLayerAddedResult inputs_layer =
              pcg_add_input_layer(pcg, input_tensor_shape);
          parallel_tensor_guid_t t_input =
              require_only_key(inputs_layer.outputs, TensorSlotName::OUTPUT);

          RepartitionAttrs repartition_attrs{ff_dim_t{nonnegative_int{0}}, 2_p};
          ParallelLayerAddedResult repartition_operator =
              add_parallel_layer(pcg,
                                 make_layer_attrs(repartition_attrs),
                                 {{TensorSlotName::INPUT, t_input}},
                                 {});
          parallel_tensor_guid_t t_repartitioned = require_only_key(
              repartition_operator.outputs, TensorSlotName::OUTPUT);

          ParallelLayerAddedResult relu_operator =
              add_parallel_layer(pcg,
                                 make_layer_attrs(make_relu_attrs()),
                                 {{TensorSlotName::INPUT, t_repartitioned}},
                                 {});
          parallel_tensor_guid_t t_relu_output =
              require_only_key(relu_operator.outputs, TensorSlotName::OUTPUT);

          MachineSpaceCoordinate cpu0{0_n, 0_n, DeviceType::CPU};
          MachineSpaceCoordinate cpu1{0_n, 1_n, DeviceType::CPU};

          ParallelTensorSpaceCoordinate tensor_coord0{0_n, 0_n, FFOrdered{0_n}};
          ParallelTensorSpaceCoordinate tensor_coord_shard0{
              0_n, 0_n, FFOrdered{0_n}};
          ParallelTensorSpaceCoordinate tensor_coord_shard1{
              0_n, 0_n, FFOrdered{1_n}};

          MappedParallelComputationGraph mpcg{
              pcg,
              {
                  {inputs_layer.parallel_layer,
                   MappedOperatorTaskGroup{
                       {{cpu0,
                         OperatorAtomicTaskShardBinding{
                             {{TensorSlotName::OUTPUT, tensor_coord0}}}}}}},
                  {repartition_operator.parallel_layer,
                   MappedOperatorTaskGroup{{
                       {cpu0,
                        OperatorAtomicTaskShardBinding{
                            {{TensorSlotName::OUTPUT, tensor_coord_shard0}}}},
                       {cpu1,
                        OperatorAtomicTaskShardBinding{
                            {{TensorSlotName::OUTPUT, tensor_coord_shard1}}}},
                   }}},
                  {relu_operator.parallel_layer,
                   MappedOperatorTaskGroup{{
                       {cpu0,
                        OperatorAtomicTaskShardBinding{{
                            {TensorSlotName::INPUT, tensor_coord_shard0},
                            {TensorSlotName::OUTPUT, tensor_coord_shard0},
                        }}},
                       {cpu1,
                        OperatorAtomicTaskShardBinding{{
                            {TensorSlotName::INPUT, tensor_coord_shard1},
                            {TensorSlotName::OUTPUT, tensor_coord_shard1},
                        }}},
                   }}},
              }};

          // build DynamicValueAttrs key for external input
          ParallelTensorAttrs input_ptensor_attrs =
              get_parallel_tensor_attrs(pcg, t_input);

          DynamicValueAttrs input_value_attrs{
              dynamic_tensor_guid_t{t_input},
              input_ptensor_attrs.shape,
              tensor_coord0,
              bidict<ParallelTensorSpaceCoordinate, MachineSpaceCoordinate>{
                  {tensor_coord0, cpu0}},
              std::nullopt,
              DynamicTensorRole{FwbTensorType::FORWARD},
          };

          std::unordered_map<DynamicValueAttrs, DynamicTensorAccessor>
              input_tensors;
          input_tensors.insert(
              {input_value_attrs, DynamicTensorAccessor{input_tensor}});

          OptimizerAttrs optimizer_attrs =
              OptimizerAttrs{SGDOptimizerAttrs{0.001, 0.9, false, 0.001}};

          DistributedFfHandle device_handle =
              create_distributed_ff_handle(ctx, 1024 * 1024, true);

          PCGInstance pcg_instance =
              create_pcg_instance(ctx,
                                  mpcg,
                                  optimizer_attrs,
                                  std::nullopt,
                                  input_tensors,
                                  ProfilingSettings{0_n, 1_p},
                                  device_handle,
                                  FFIterationConfig{1_p});

          perform_all_passes_for_pcg_instance(pcg_instance,
                                              ProfilingSettings{0_n, 1_p},
                                              device_handle,
                                              FFIterationConfig{1_p});

          ctx.get_outstanding_events().wait();

          TensorInstanceBacking const &backing =
              pcg_instance.get_tensor_instance_backing();

          ParallelTensorAttrs relu_output_attrs =
              get_parallel_tensor_attrs(pcg, t_relu_output);

          // verify both relu output shards
          auto make_relu_key =
              [&](ParallelTensorSpaceCoordinate const &coord,
                  MachineSpaceCoordinate const &machine) -> DynamicValueAttrs {
            return DynamicValueAttrs{
                dynamic_tensor_guid_t{t_relu_output},
                relu_output_attrs.shape,
                coord,
                bidict<ParallelTensorSpaceCoordinate, MachineSpaceCoordinate>{
                    {coord, machine}},
                std::nullopt,
                DynamicTensorRole{FwbTensorType::FORWARD},
            };
          };

          auto [relu0_inst, relu0_ready] =
              backing.backing.at(make_relu_key(tensor_coord_shard0, cpu0));
          auto [relu1_inst, relu1_ready] =
              backing.backing.at(make_relu_key(tensor_coord_shard1, cpu1));

          GenericTensorAccessorR relu0_accessor =
              dynamic_tensor_accessor_from_instance(relu0_inst,
                                                    relu0_ready,
                                                    relu_output_attrs.shape,
                                                    Permissions::RO,
                                                    ctx.get_current_processor())
                  .get<GenericTensorAccessorR>();

          GenericTensorAccessorR relu1_accessor =
              dynamic_tensor_accessor_from_instance(relu1_inst,
                                                    relu1_ready,
                                                    relu_output_attrs.shape,
                                                    Permissions::RO,
                                                    ctx.get_current_processor())
                  .get<GenericTensorAccessorR>();

          // repartition splits along dim0 (batch) with degree 2
          // in Fortran order (dim0 fastest):
          // shard0 covers rows [0..4]: ptr[0]=0, ptr[1]=1, ..., ptr[4]=4
          //                            ptr[5]=10, ptr[6]=11, ... (col 1)
          // shard1 covers rows [5..9]: ptr[0]=5, ptr[1]=6, ..., ptr[4]=9
          //                            ptr[5]=15, ptr[6]=16, ... (col 1)
          // all values non-negative so relu doesn't change them

          float const *relu0_ptr = relu0_accessor.get_float_ptr();
          float const *relu1_ptr = relu1_accessor.get_float_ptr();

          // shard0: rows 0-4 of input
          // in Fortran order: ptr[i*5 + j]... actually
          // shard0 instance rect [0..4, 0..15] in Fortran order:
          // ptr[0]=input[0,0]=0, ptr[1]=input[1,0]=1, ..., ptr[4]=input[4,0]=4
          // ptr[5]=input[0,1]=10, ptr[6]=input[1,1]=11, ...
          int shard_size = (batch_size.int_from_positive_int() / 2) *
                           data_dim.int_from_positive_int();

          for (int row = 0; row < batch_size.int_from_positive_int() / 2;
               row++) {
            for (int col = 0; col < data_dim.int_from_positive_int(); col++) {
              // Fortran order: flat_idx = row + col * (batch/2)
              int flat_idx =
                  row + col * (batch_size.int_from_positive_int() / 2);
              // shard0: actual row in full tensor = row (0..4)
              float expected0 = static_cast<float>(
                  row + col * batch_size.int_from_positive_int());
              // shard1: actual row in full tensor = row + 5 (5..9)
              float expected1 = static_cast<float>(
                  (row + batch_size.int_from_positive_int() / 2) +
                  col * batch_size.int_from_positive_int());
              INFO("row=", row, " col=", col, " flat_idx=", flat_idx);
              CHECK_EQ(relu0_ptr[flat_idx], expected0);
              CHECK_EQ(relu1_ptr[flat_idx], expected1);
            }
          }
        });
    result.wait();
  }
}
TEST_SUITE(FF_CUDA_TEST_SUITE) {
  TEST_CASE(
      "RealmBackend e2e Training Repartition Op (GPU Model Parallelism)") {
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

          ParallelLayerAddedResult inputs_layer =
              pcg_add_input_layer(pcg, input_tensor_shape);
          parallel_tensor_guid_t t_input =
              require_only_key(inputs_layer.outputs, TensorSlotName::OUTPUT);

          // repartition along batch dimension (dim 0) with degree 2
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

          ParallelLayerAddedResult relu_operator =
              add_parallel_layer(pcg,
                                 make_layer_attrs(make_relu_attrs()),
                                 {{TensorSlotName::INPUT, t_repartitioned}},
                                 /*weights=*/{});

          MachineSpaceCoordinate gpu0{0_n, 0_n, DeviceType::GPU};
          MachineSpaceCoordinate gpu1{0_n, 1_n, DeviceType::GPU};

          // input: one shard on gpu0 (not yet repartitioned)
          ParallelTensorSpaceCoordinate tensor_coord0{0_n, 0_n, FFOrdered{0_n}};
          // after repartition: two shards along dim 0
          ParallelTensorSpaceCoordinate tensor_coord_shard0{
              0_n, 0_n, FFOrdered{0_n}};
          ParallelTensorSpaceCoordinate tensor_coord_shard1{
              0_n, 0_n, FFOrdered{1_n}};

          MappedParallelComputationGraph mpcg{
              pcg,
              {
                  {inputs_layer.parallel_layer,
                   MappedOperatorTaskGroup{
                       {{gpu0,
                         OperatorAtomicTaskShardBinding{
                             {{TensorSlotName::OUTPUT, tensor_coord0}}}}}}},
                  // repartition: OUTPUT only (no INPUT in binding)
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
                  {relu_operator.parallel_layer,
                   MappedOperatorTaskGroup{{
                       {gpu0,
                        OperatorAtomicTaskShardBinding{{
                            {TensorSlotName::INPUT, tensor_coord_shard0},
                            {TensorSlotName::OUTPUT, tensor_coord_shard0},
                        }}},
                       {gpu1,
                        OperatorAtomicTaskShardBinding{{
                            {TensorSlotName::INPUT, tensor_coord_shard1},
                            {TensorSlotName::OUTPUT, tensor_coord_shard1},
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
                                  ProfilingSettings{0_n, 1_p},
                                  device_handle,
                                  FFIterationConfig{1_p});

          perform_all_passes_for_pcg_instance(pcg_instance,
                                              ProfilingSettings{0_n, 1_p},
                                              device_handle,
                                              FFIterationConfig{1_p});
        });
    result.wait();
  }
}
} // namespace test
