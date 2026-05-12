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
#include "realm-execution/external_tensor_binding.h"
#include "realm-execution/external_tensor_handle.h"
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
  TEST_CASE("RealmBackend Replicate Op with External Input Instance (CPU)") {
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

          // allocate external input tensor and fill with known values
          GenericTensorAccessorW input_tensor =
              allocator.allocate_tensor(input_tensor_shape);
          float *input_ptr = input_tensor.get_float_ptr();
          int num_elements = batch_size.int_from_positive_int() *
                             data_dim.int_from_positive_int();

          for (int i = 0; i < num_elements; i++) {
            input_ptr[i] = static_cast<float>(i);
          }
          // construct PCG
          ParallelComputationGraph pcg = empty_parallel_computation_graph();

          ParallelLayerAddedResult inputs_layer =
              pcg_add_input_layer(pcg, input_tensor_shape);
          parallel_tensor_guid_t t_input =
              require_only_key(inputs_layer.outputs, TensorSlotName::OUTPUT);

          ReplicateAttrs repl_attrs{/*replicate_degree=*/2_p};
          ParallelLayerAddedResult repl_operator =
              add_parallel_layer(pcg,
                                 make_layer_attrs(repl_attrs),
                                 {{TensorSlotName::INPUT, t_input}},
                                 /*weights=*/{});
          parallel_tensor_guid_t t_repl =
              require_only_key(repl_operator.outputs, TensorSlotName::OUTPUT);

          ParallelLayerAddedResult relu_operator =
              add_parallel_layer(pcg,
                                 make_layer_attrs(make_relu_attrs()),
                                 {{TensorSlotName::INPUT, t_repl}},
                                 /*weights=*/{});

          MachineSpaceCoordinate cpu0{0_n, 0_n, DeviceType::CPU};
          MachineSpaceCoordinate cpu1{0_n, 1_n, DeviceType::CPU};

          ParallelTensorSpaceCoordinate tensor_coord0{0_n, 0_n, FFOrdered{0_n}};
          ParallelTensorSpaceCoordinate tensor_coord1{0_n, 1_n, FFOrdered{0_n}};

          MappedParallelComputationGraph mpcg{
              pcg,
              {
                  {inputs_layer.parallel_layer,
                   MappedOperatorTaskGroup{
                       {{cpu0,
                         OperatorAtomicTaskShardBinding{
                             {{TensorSlotName::OUTPUT, tensor_coord0}}}}}}},
                  {repl_operator.parallel_layer,
                   MappedOperatorTaskGroup{{
                       {cpu0,
                        OperatorAtomicTaskShardBinding{{
                            {TensorSlotName::OUTPUT, tensor_coord0},
                        }}},
                       {cpu1,
                        OperatorAtomicTaskShardBinding{{
                            {TensorSlotName::OUTPUT, tensor_coord1},
                        }}},
                   }}},
                  {relu_operator.parallel_layer,
                   MappedOperatorTaskGroup{{
                       {cpu0,
                        OperatorAtomicTaskShardBinding{{
                            {TensorSlotName::INPUT, tensor_coord0},
                            {TensorSlotName::OUTPUT, tensor_coord0},
                        }}},
                       {cpu1,
                        OperatorAtomicTaskShardBinding{{
                            {TensorSlotName::INPUT, tensor_coord1},
                            {TensorSlotName::OUTPUT, tensor_coord1},
                        }}},
                   }}},
              }};

          // build DynamicValueAttrs key for the input tensor
          // must match exactly what make_dynamic_open_dataflow_graph produces
          ParallelTensorAttrs input_ptensor_attrs =
              get_parallel_tensor_attrs(pcg, t_input);

          bidict<ParallelTensorSpaceCoordinate, MachineSpaceCoordinate>
              input_mapping{{tensor_coord0, cpu0}};

          DynamicValueAttrs input_value_attrs{
              /*tensor_guid=*/dynamic_tensor_guid_t{t_input},
              /*parallel_tensor_shape=*/input_ptensor_attrs.shape,
              /*shard_coord=*/tensor_coord0,
              /*mapping=*/input_mapping,
              /*accessor=*/std::nullopt,
              /*role=*/DynamicTensorRole{FwbTensorType::FORWARD},
          };

          // pass external tensor as preallocated input
          std::unordered_map<DynamicValueAttrs, DynamicTensorAccessor>
              input_tensors;
          input_tensors.insert(
              {input_value_attrs, DynamicTensorAccessor{input_tensor}});

          OptimizerAttrs optimizer_attrs =
              OptimizerAttrs{SGDOptimizerAttrs{/*lr=*/0.001,
                                               /*momentum=*/0.9,
                                               /*nesterov=*/false,
                                               /*weight_decay=*/0.001}};

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

          // wait for ALL outstanding Realm events (copies, tasks, reductions)
          // to complete before reading back tensor values
          ctx.get_outstanding_events().wait();

          parallel_tensor_guid_t t_relu_output =
              require_only_key(relu_operator.outputs, TensorSlotName::OUTPUT);

          ParallelTensorAttrs relu_output_attrs =
              get_parallel_tensor_attrs(pcg, t_relu_output);

          auto make_output_key =
              [&](parallel_tensor_guid_t guid,
                  ParallelTensorAttrs const &attrs,
                  ParallelTensorSpaceCoordinate const &coord,
                  MachineSpaceCoordinate const &machine) -> DynamicValueAttrs {
            return DynamicValueAttrs{
                /*tensor_guid=*/dynamic_tensor_guid_t{guid},
                /*parallel_tensor_shape=*/attrs.shape,
                /*shard_coord=*/coord,
                /*mapping=*/
                bidict<ParallelTensorSpaceCoordinate, MachineSpaceCoordinate>{
                    {coord, machine}},
                /*accessor=*/std::nullopt,
                /*role=*/DynamicTensorRole{FwbTensorType::FORWARD},
            };
          };

          DynamicValueAttrs relu0_key = make_output_key(
              t_relu_output, relu_output_attrs, tensor_coord0, cpu0);
          DynamicValueAttrs relu1_key = make_output_key(
              t_relu_output, relu_output_attrs, tensor_coord1, cpu1);

          // get tensor instance backing
          TensorInstanceBacking const &backing =
              pcg_instance.get_tensor_instance_backing();

          auto [relu0_inst, relu0_ready] = backing.backing.at(relu0_key);
          auto [relu1_inst, relu1_ready] = backing.backing.at(relu1_key);

          // convert to accessors — events already waited above
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

          // verify replica0 == replica1
          CHECK(tensor_accessor_all(compare_tensor_accessors_eq(
              relu0_accessor, relu1_accessor, allocator)));
          // verify values match input — input was 0,1,...,159
          // all non-negative so relu doesn't change them
          float const *relu0_ptr = relu0_accessor.get_float_ptr();
          for (int i = 0; i < num_elements; i++) {
            CHECK_EQ(relu0_ptr[i], static_cast<float>(i));
          }
        });
    result.wait();
  }
}

TEST_SUITE(FF_CUDA_TEST_SUITE) {
  TEST_CASE("RealmBackend Replicate Op with External Input Instance (GPU)") {
    std::vector<char *> fake_args =
        make_fake_realm_args(/*num_cpus=*/1_p, /*num_gpus=*/2_n);
    int fake_argc = fake_args.size();
    char **fake_argv = fake_args.data();

    RealmManager manager = RealmManager{&fake_argc, &fake_argv};
    ControllerTaskResult result =
        manager.start_controller([](RealmContext &ctx) {
          positive_int batch_size = 10_p;
          positive_int data_dim = 16_p;
          int num_elements = batch_size.int_from_positive_int() *
                             data_dim.int_from_positive_int();

          TensorShape input_tensor_shape = TensorShape{
              TensorDims{FFOrdered{batch_size, data_dim}}, DataType::FLOAT};

          MachineSpaceCoordinate gpu0{0_n, 0_n, DeviceType::GPU};
          MachineSpaceCoordinate gpu1{0_n, 1_n, DeviceType::GPU};

          // create external tensor in CPU mem
          // accessible from GPU
          ExternalTensorHandle input_handle =
              ctx.create_external_tensor(gpu0, input_tensor_shape);

          // fill with known values from CPU
          float *ptr = input_handle.get_float_ptr();
          for (int i = 0; i < num_elements; i++) {
            ptr[i] = static_cast<float>(i);
          }

          // construct PCG
          ParallelComputationGraph pcg = empty_parallel_computation_graph();

          ParallelLayerAddedResult inputs_layer =
              pcg_add_input_layer(pcg, input_tensor_shape);
          parallel_tensor_guid_t t_input =
              require_only_key(inputs_layer.outputs, TensorSlotName::OUTPUT);

          ReplicateAttrs repl_attrs{2_p};
          ParallelLayerAddedResult repl_operator =
              add_parallel_layer(pcg,
                                 make_layer_attrs(repl_attrs),
                                 {{TensorSlotName::INPUT, t_input}},
                                 {});
          parallel_tensor_guid_t t_repl =
              require_only_key(repl_operator.outputs, TensorSlotName::OUTPUT);

          ParallelLayerAddedResult relu_operator =
              add_parallel_layer(pcg,
                                 make_layer_attrs(make_relu_attrs()),
                                 {{TensorSlotName::INPUT, t_repl}},
                                 {});
          parallel_tensor_guid_t t_relu_output =
              require_only_key(relu_operator.outputs, TensorSlotName::OUTPUT);

          ParallelTensorSpaceCoordinate tensor_coord0{0_n, 0_n, FFOrdered{0_n}};
          ParallelTensorSpaceCoordinate tensor_coord1{0_n, 1_n, FFOrdered{0_n}};

          MappedParallelComputationGraph mpcg{
              pcg,
              {
                  {inputs_layer.parallel_layer,
                   MappedOperatorTaskGroup{
                       {{gpu0,
                         OperatorAtomicTaskShardBinding{
                             {{TensorSlotName::OUTPUT, tensor_coord0}}}}}}},
                  {repl_operator.parallel_layer,
                   MappedOperatorTaskGroup{{
                       {gpu0,
                        OperatorAtomicTaskShardBinding{
                            {{TensorSlotName::OUTPUT, tensor_coord0}}}},
                       {gpu1,
                        OperatorAtomicTaskShardBinding{
                            {{TensorSlotName::OUTPUT, tensor_coord1}}}},
                   }}},
                  {relu_operator.parallel_layer,
                   MappedOperatorTaskGroup{{
                       {gpu0,
                        OperatorAtomicTaskShardBinding{{
                            {TensorSlotName::INPUT, tensor_coord0},
                            {TensorSlotName::OUTPUT, tensor_coord0},
                        }}},
                       {gpu1,
                        OperatorAtomicTaskShardBinding{{
                            {TensorSlotName::INPUT, tensor_coord1},
                            {TensorSlotName::OUTPUT, tensor_coord1},
                        }}},
                   }}},
              }};

          OptimizerAttrs optimizer_attrs =
              OptimizerAttrs{SGDOptimizerAttrs{0.001, 0.9, false, 0.001}};

          DistributedFfHandle device_handle =
              create_distributed_ff_handle(ctx, 1024 * 1024, true);

          PCGInstance pcg_instance =
              create_pcg_instance(ctx,
                                  mpcg,
                                  optimizer_attrs,
                                  std::nullopt,
                                  {}, // no DynamicTensorAccessor inputs
                                  ProfilingSettings{0_n, 1_p},
                                  device_handle,
                                  FFIterationConfig{1_p},
                                  {ExternalTensorBinding{
                                      /*tensor_guid=*/t_input,
                                      /*shard_coord=*/tensor_coord0,
                                      /*machine_coord=*/gpu0,
                                      /*handle=*/input_handle,
                                  }});

          perform_all_passes_for_pcg_instance(pcg_instance,
                                              ProfilingSettings{0_n, 1_p},
                                              device_handle,
                                              FFIterationConfig{1_p});

          ctx.get_outstanding_events().wait();

          // verify relu output on both GPUs
          TensorInstanceBacking const &backing =
              pcg_instance.get_tensor_instance_backing();

          ParallelTensorAttrs relu_output_attrs =
              get_parallel_tensor_attrs(pcg, t_relu_output);

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
              backing.backing.at(make_relu_key(tensor_coord0, gpu0));
          auto [relu1_inst, relu1_ready] =
              backing.backing.at(make_relu_key(tensor_coord1, gpu1));

          // copy GPU tensors to CPU for verification
          Allocator cpu_allocator = ctx.get_current_device_allocator();
          GenericTensorAccessorR relu0_cpu = ctx.copy_instance_to_cpu(
              relu0_inst, relu0_ready, relu_output_attrs.shape);

          GenericTensorAccessorR relu1_cpu = ctx.copy_instance_to_cpu(
              relu1_inst, relu1_ready, relu_output_attrs.shape);
          // both replicas should match input — all non-negative so relu
          // doesn't change values
          CHECK(tensor_accessor_all(compare_tensor_accessors_eq(
              relu0_cpu, relu1_cpu, cpu_allocator)));

          float const *relu0_ptr = relu0_cpu.get_float_ptr();
          for (int i = 0; i < num_elements; i++) {
            INFO("index = ", i);
            CHECK_EQ(relu0_ptr[i], static_cast<float>(i));
          }
        });
    result.wait();
  }
}
} // namespace test
