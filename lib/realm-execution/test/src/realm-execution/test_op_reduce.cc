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
  TEST_CASE("RealmBackend e2e Training Reduction Op with External Instances "
            "(CPU Model Parallelism)") {
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

      // allocate external input tensor — fill with 1s
      GenericTensorAccessorW input_tensor =
          allocator.allocate_tensor(input_tensor_shape);
      float *input_ptr = input_tensor.get_float_ptr();
      int input_num_elements = batch_size.int_from_positive_int() *
                               in_channels.int_from_positive_int();
      for (int i = 0; i < input_num_elements; i++) {
        input_ptr[i] = 1.0f;
      }

      // allocate external weight tensor — fill with 1s
      GenericTensorAccessorW weight_tensor =
          allocator.allocate_tensor(weight_tensor_shape);
      float *weight_ptr = weight_tensor.get_float_ptr();
      int weight_num_elements = out_channels.int_from_positive_int() *
                                in_channels.int_from_positive_int();
      for (int i = 0; i < weight_num_elements; i++) {
        weight_ptr[i] = 1.0f;
      }

      // ... PCG construction (same as existing test) ...
      ParallelComputationGraph pcg = empty_parallel_computation_graph();

      ParallelLayerAddedResult inputs_layer =
          pcg_add_input_layer(pcg, input_tensor_shape);
      parallel_tensor_guid_t t_input =
          require_only_key(inputs_layer.outputs, TensorSlotName::OUTPUT);

      ParallelLayerAddedResult weights_layer =
          pcg_add_input_layer(pcg, weight_tensor_shape);
      parallel_tensor_guid_t t_weight =
          require_only_key(weights_layer.outputs, TensorSlotName::OUTPUT);

      RepartitionAttrs input_repartition_attrs{ff_dim_t{nonnegative_int{1}},
                                               2_p};
      ParallelLayerAddedResult input_repartition_operator =
          add_parallel_layer(pcg,
                             make_layer_attrs(input_repartition_attrs),
                             {{TensorSlotName::INPUT, t_input}},
                             {});
      parallel_tensor_guid_t t_input_repartitioned = require_only_key(
          input_repartition_operator.outputs, TensorSlotName::OUTPUT);

      RepartitionAttrs weight_repartition_attrs{ff_dim_t{nonnegative_int{1}},
                                                2_p};
      ParallelLayerAddedResult weight_repartition_operator =
          add_parallel_layer(pcg,
                             make_layer_attrs(weight_repartition_attrs),
                             {{TensorSlotName::INPUT, t_weight}},
                             {});
      parallel_tensor_guid_t t_weight_repartitioned = require_only_key(
          weight_repartition_operator.outputs, TensorSlotName::OUTPUT);

      ParallelLayerAddedResult linear_operator = add_parallel_layer(
          pcg,
          ParallelLayerAttrs{PCGOperatorAttrs{LinearAttrs{out_channels,
                                                          false,
                                                          DataType::FLOAT,
                                                          Activation::RELU,
                                                          std::nullopt}},
                             std::nullopt},
          {{TensorSlotName::INPUT, t_input_repartitioned}},
          {{TensorSlotName::WEIGHT, t_weight_repartitioned}});
      parallel_tensor_guid_t t_linear =
          require_only_key(linear_operator.outputs, TensorSlotName::OUTPUT);

      ReductionAttrs reduction_attrs{2_p};
      ParallelLayerAddedResult reduction_operator =
          add_parallel_layer(pcg,
                             make_layer_attrs(reduction_attrs),
                             {{TensorSlotName::INPUT, t_linear}},
                             {});
      parallel_tensor_guid_t t_reduced =
          require_only_key(reduction_operator.outputs, TensorSlotName::OUTPUT);

      ParallelLayerAddedResult relu_operator =
          add_parallel_layer(pcg,
                             make_layer_attrs(make_relu_attrs()),
                             {{TensorSlotName::INPUT, t_reduced}},
                             {});
      parallel_tensor_guid_t t_relu_output =
          require_only_key(relu_operator.outputs, TensorSlotName::OUTPUT);

      MachineSpaceCoordinate cpu0{0_n, 0_n, DeviceType::CPU};
      MachineSpaceCoordinate cpu1{0_n, 1_n, DeviceType::CPU};

      ParallelTensorSpaceCoordinate input_coord{0_n, 0_n, FFOrdered{0_n, 0_n}};
      ParallelTensorSpaceCoordinate weight_coord{0_n, 0_n, FFOrdered{0_n, 0_n}};
      ParallelTensorSpaceCoordinate input_repartitioned_coord_0{
          0_n, 0_n, FFOrdered{0_n, 0_n}};
      ParallelTensorSpaceCoordinate input_repartitioned_coord_1{
          0_n, 0_n, FFOrdered{0_n, 1_n}};
      ParallelTensorSpaceCoordinate weight_repartitioned_coord_0{
          0_n, 0_n, FFOrdered{0_n, 0_n}};
      ParallelTensorSpaceCoordinate weight_repartitioned_coord_1{
          0_n, 0_n, FFOrdered{0_n, 1_n}};
      ParallelTensorSpaceCoordinate linear_coord_0{
          0_n, 0_n, FFOrdered{0_n, 0_n}};
      ParallelTensorSpaceCoordinate linear_coord_1{
          1_n, 0_n, FFOrdered{0_n, 0_n}};
      ParallelTensorSpaceCoordinate reduced_coord{
          0_n, 0_n, FFOrdered{0_n, 0_n}};

      MappedParallelComputationGraph mpcg{
          pcg,
          {
              {inputs_layer.parallel_layer,
               MappedOperatorTaskGroup{
                   {{cpu0,
                     OperatorAtomicTaskShardBinding{
                         {{TensorSlotName::OUTPUT, input_coord}}}}}}},
              {weights_layer.parallel_layer,
               MappedOperatorTaskGroup{
                   {{cpu0,
                     OperatorAtomicTaskShardBinding{
                         {{TensorSlotName::OUTPUT, weight_coord}}}}}}},
              {input_repartition_operator.parallel_layer,
               MappedOperatorTaskGroup{{
                   {cpu0,
                    OperatorAtomicTaskShardBinding{
                        {{TensorSlotName::OUTPUT,
                          input_repartitioned_coord_0}}}},
                   {cpu1,
                    OperatorAtomicTaskShardBinding{
                        {{TensorSlotName::OUTPUT,
                          input_repartitioned_coord_1}}}},
               }}},
              {weight_repartition_operator.parallel_layer,
               MappedOperatorTaskGroup{{
                   {cpu0,
                    OperatorAtomicTaskShardBinding{
                        {{TensorSlotName::OUTPUT,
                          weight_repartitioned_coord_0}}}},
                   {cpu1,
                    OperatorAtomicTaskShardBinding{
                        {{TensorSlotName::OUTPUT,
                          weight_repartitioned_coord_1}}}},
               }}},
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
              {reduction_operator.parallel_layer,
               MappedOperatorTaskGroup{{
                   {cpu0,
                    OperatorAtomicTaskShardBinding{
                        {{TensorSlotName::INPUT, linear_coord_0}}}},
                   {cpu1,
                    OperatorAtomicTaskShardBinding{
                        {{TensorSlotName::INPUT, linear_coord_1}}}},
               }}},
              {relu_operator.parallel_layer,
               MappedOperatorTaskGroup{
                   {{cpu0,
                     OperatorAtomicTaskShardBinding{{
                         {TensorSlotName::INPUT, reduced_coord},
                         {TensorSlotName::OUTPUT, reduced_coord},
                     }}}}}},
          }};

      // build DynamicValueAttrs keys for external inputs
      ParallelTensorAttrs input_ptensor_attrs =
          get_parallel_tensor_attrs(pcg, t_input);
      ParallelTensorAttrs weight_ptensor_attrs =
          get_parallel_tensor_attrs(pcg, t_weight);

      DynamicValueAttrs input_value_attrs{
          dynamic_tensor_guid_t{t_input},
          input_ptensor_attrs.shape,
          input_coord,
          bidict<ParallelTensorSpaceCoordinate, MachineSpaceCoordinate>{
              {input_coord, cpu0}},
          std::nullopt,
          DynamicTensorRole{FwbTensorType::FORWARD},
      };

      DynamicValueAttrs weight_value_attrs{
          dynamic_tensor_guid_t{t_weight},
          weight_ptensor_attrs.shape,
          weight_coord,
          bidict<ParallelTensorSpaceCoordinate, MachineSpaceCoordinate>{
              {weight_coord, cpu0}},
          std::nullopt,
          DynamicTensorRole{FwbTensorType::FORWARD},
      };

      std::unordered_map<DynamicValueAttrs, DynamicTensorAccessor>
          input_tensors;
      input_tensors.insert(
          {input_value_attrs, DynamicTensorAccessor{input_tensor}});
      input_tensors.insert(
          {weight_value_attrs, DynamicTensorAccessor{weight_tensor}});

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

      // wait for all outstanding events
      ctx.get_outstanding_events().wait();

      // verify relu output
      TensorInstanceBacking const &backing =
          pcg_instance.get_tensor_instance_backing();

      ParallelTensorAttrs relu_output_attrs =
          get_parallel_tensor_attrs(pcg, t_relu_output);

      DynamicValueAttrs relu_output_key{
          dynamic_tensor_guid_t{t_relu_output},
          relu_output_attrs.shape,
          reduced_coord,
          bidict<ParallelTensorSpaceCoordinate, MachineSpaceCoordinate>{
              {reduced_coord, cpu0}},
          std::nullopt,
          DynamicTensorRole{FwbTensorType::FORWARD},
      };

      auto [relu_inst, relu_ready] = backing.backing.at(relu_output_key);

      GenericTensorAccessorR relu_accessor =
          dynamic_tensor_accessor_from_instance(relu_inst,
                                                relu_ready,
                                                relu_output_attrs.shape,
                                                Permissions::RO,
                                                ctx.get_current_processor())
              .get<GenericTensorAccessorR>();

      // each shard has input[4,4] @ weight[4,4].T
      // = sum of 4 ones = 4.0 per element
      // relu(4.0) = 4.0
      // reduction sums 2 shards: 4.0 + 4.0 = 8.0
      // relu(8.0) = 8.0
      float const *relu_ptr = relu_accessor.get_float_ptr();
      int output_num_elements = batch_size.int_from_positive_int() *
                                out_channels.int_from_positive_int();
      for (int i = 0; i < output_num_elements; i++) {
        CHECK_EQ(relu_ptr[i], 8.0f);
      }
    });
    result.wait();
  }
}

TEST_SUITE(FF_CUDA_TEST_SUITE) {
  TEST_CASE("RealmBackend Reduction Op with External Input Instance (GPU)") {
    std::vector<char *> fake_args =
        make_fake_realm_args(/*num_cpus=*/1_p, /*num_gpus=*/2_n);
    int fake_argc = fake_args.size();
    char **fake_argv = fake_args.data();

    RealmManager manager = RealmManager{&fake_argc, &fake_argv};
    ControllerTaskResult result = manager.start_controller([](RealmContext
                                                                  &ctx) {
      positive_int batch_size = 4_p;
      positive_int in_channels = 8_p;
      positive_int out_channels = 4_p;

      TensorShape input_tensor_shape = TensorShape{
          TensorDims{FFOrdered{batch_size, in_channels}}, DataType::FLOAT};
      TensorShape weight_tensor_shape = TensorShape{
          TensorDims{FFOrdered{out_channels, in_channels}}, DataType::FLOAT};

      MachineSpaceCoordinate gpu0{0_n, 0_n, DeviceType::GPU};
      MachineSpaceCoordinate gpu1{0_n, 1_n, DeviceType::GPU};

      // create external tensors
      ExternalTensorHandle input_handle =
          ctx.create_external_tensor(gpu0, input_tensor_shape);
      ExternalTensorHandle weight_handle =
          ctx.create_external_tensor(gpu0, weight_tensor_shape);

      // fill with 1s
      int input_num_elements = batch_size.int_from_positive_int() *
                               in_channels.int_from_positive_int();
      int weight_num_elements = out_channels.int_from_positive_int() *
                                in_channels.int_from_positive_int();

      float *input_ptr = input_handle.get_float_ptr();
      for (int i = 0; i < input_num_elements; i++) {
        input_ptr[i] = 1.0f;
      }
      float *weight_ptr = weight_handle.get_float_ptr();
      for (int i = 0; i < weight_num_elements; i++) {
        weight_ptr[i] = 1.0f;
      }

      // PCG: same as existing reduction test
      ParallelComputationGraph pcg = empty_parallel_computation_graph();

      ParallelLayerAddedResult inputs_layer =
          pcg_add_input_layer(pcg, input_tensor_shape);
      parallel_tensor_guid_t t_input =
          require_only_key(inputs_layer.outputs, TensorSlotName::OUTPUT);

      ParallelLayerAddedResult weights_layer =
          pcg_add_input_layer(pcg, weight_tensor_shape);
      parallel_tensor_guid_t t_weight =
          require_only_key(weights_layer.outputs, TensorSlotName::OUTPUT);

      RepartitionAttrs input_repartition_attrs{ff_dim_t{nonnegative_int{1}},
                                               2_p};
      ParallelLayerAddedResult input_repartition_operator =
          add_parallel_layer(pcg,
                             make_layer_attrs(input_repartition_attrs),
                             {{TensorSlotName::INPUT, t_input}},
                             {});
      parallel_tensor_guid_t t_input_repartitioned = require_only_key(
          input_repartition_operator.outputs, TensorSlotName::OUTPUT);

      RepartitionAttrs weight_repartition_attrs{ff_dim_t{nonnegative_int{1}},
                                                2_p};
      ParallelLayerAddedResult weight_repartition_operator =
          add_parallel_layer(pcg,
                             make_layer_attrs(weight_repartition_attrs),
                             {{TensorSlotName::INPUT, t_weight}},
                             {});
      parallel_tensor_guid_t t_weight_repartitioned = require_only_key(
          weight_repartition_operator.outputs, TensorSlotName::OUTPUT);

      ParallelLayerAddedResult linear_operator = add_parallel_layer(
          pcg,
          ParallelLayerAttrs{PCGOperatorAttrs{LinearAttrs{out_channels,
                                                          false,
                                                          DataType::FLOAT,
                                                          Activation::RELU,
                                                          std::nullopt}},
                             std::nullopt},
          {{TensorSlotName::INPUT, t_input_repartitioned}},
          {{TensorSlotName::WEIGHT, t_weight_repartitioned}});
      parallel_tensor_guid_t t_linear =
          require_only_key(linear_operator.outputs, TensorSlotName::OUTPUT);

      ReductionAttrs reduction_attrs{2_p};
      ParallelLayerAddedResult reduction_operator =
          add_parallel_layer(pcg,
                             make_layer_attrs(reduction_attrs),
                             {{TensorSlotName::INPUT, t_linear}},
                             {});
      parallel_tensor_guid_t t_reduced =
          require_only_key(reduction_operator.outputs, TensorSlotName::OUTPUT);

      ParallelLayerAddedResult relu_operator =
          add_parallel_layer(pcg,
                             make_layer_attrs(make_relu_attrs()),
                             {{TensorSlotName::INPUT, t_reduced}},
                             {});
      parallel_tensor_guid_t t_relu_output =
          require_only_key(relu_operator.outputs, TensorSlotName::OUTPUT);

      // coords — same as existing reduction test
      ParallelTensorSpaceCoordinate input_coord{0_n, 0_n, FFOrdered{0_n, 0_n}};
      ParallelTensorSpaceCoordinate weight_coord{0_n, 0_n, FFOrdered{0_n, 0_n}};
      ParallelTensorSpaceCoordinate input_repartitioned_coord_0{
          0_n, 0_n, FFOrdered{0_n, 0_n}};
      ParallelTensorSpaceCoordinate input_repartitioned_coord_1{
          0_n, 0_n, FFOrdered{0_n, 1_n}};
      ParallelTensorSpaceCoordinate weight_repartitioned_coord_0{
          0_n, 0_n, FFOrdered{0_n, 0_n}};
      ParallelTensorSpaceCoordinate weight_repartitioned_coord_1{
          0_n, 0_n, FFOrdered{0_n, 1_n}};
      ParallelTensorSpaceCoordinate linear_coord_0{
          0_n, 0_n, FFOrdered{0_n, 0_n}};
      ParallelTensorSpaceCoordinate linear_coord_1{
          1_n, 0_n, FFOrdered{0_n, 0_n}};
      ParallelTensorSpaceCoordinate reduced_coord{
          0_n, 0_n, FFOrdered{0_n, 0_n}};

      MappedParallelComputationGraph mpcg{
          pcg,
          {
              {inputs_layer.parallel_layer,
               MappedOperatorTaskGroup{
                   {{gpu0,
                     OperatorAtomicTaskShardBinding{
                         {{TensorSlotName::OUTPUT, input_coord}}}}}}},
              {weights_layer.parallel_layer,
               MappedOperatorTaskGroup{
                   {{gpu0,
                     OperatorAtomicTaskShardBinding{
                         {{TensorSlotName::OUTPUT, weight_coord}}}}}}},
              {input_repartition_operator.parallel_layer,
               MappedOperatorTaskGroup{{
                   {gpu0,
                    OperatorAtomicTaskShardBinding{
                        {{TensorSlotName::OUTPUT,
                          input_repartitioned_coord_0}}}},
                   {gpu1,
                    OperatorAtomicTaskShardBinding{
                        {{TensorSlotName::OUTPUT,
                          input_repartitioned_coord_1}}}},
               }}},
              {weight_repartition_operator.parallel_layer,
               MappedOperatorTaskGroup{{
                   {gpu0,
                    OperatorAtomicTaskShardBinding{
                        {{TensorSlotName::OUTPUT,
                          weight_repartitioned_coord_0}}}},
                   {gpu1,
                    OperatorAtomicTaskShardBinding{
                        {{TensorSlotName::OUTPUT,
                          weight_repartitioned_coord_1}}}},
               }}},
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
              {reduction_operator.parallel_layer,
               MappedOperatorTaskGroup{{
                   {gpu0,
                    OperatorAtomicTaskShardBinding{
                        {{TensorSlotName::INPUT, linear_coord_0}}}},
                   {gpu1,
                    OperatorAtomicTaskShardBinding{
                        {{TensorSlotName::INPUT, linear_coord_1}}}},
               }}},
              {relu_operator.parallel_layer,
               MappedOperatorTaskGroup{
                   {{gpu0,
                     OperatorAtomicTaskShardBinding{{
                         {TensorSlotName::INPUT, reduced_coord},
                         {TensorSlotName::OUTPUT, reduced_coord},
                     }}}}}},
          }};

      OptimizerAttrs optimizer_attrs =
          OptimizerAttrs{SGDOptimizerAttrs{0.001, 0.9, false, 0.001}};

      DistributedFfHandle device_handle =
          create_distributed_ff_handle(ctx, 1024 * 1024, true);

      PCGInstance pcg_instance = create_pcg_instance(
          ctx,
          mpcg,
          optimizer_attrs,
          std::nullopt,
          {},
          ProfilingSettings{0_n, 1_p},
          device_handle,
          FFIterationConfig{1_p},
          {
              ExternalTensorBinding{t_input, input_coord, gpu0, input_handle},
              ExternalTensorBinding{
                  t_weight, weight_coord, gpu0, weight_handle},
          });

      perform_all_passes_for_pcg_instance(pcg_instance,
                                          ProfilingSettings{0_n, 1_p},
                                          device_handle,
                                          FFIterationConfig{1_p});

      ctx.get_outstanding_events().wait();

      TensorInstanceBacking const &backing =
          pcg_instance.get_tensor_instance_backing();

      ParallelTensorAttrs relu_output_attrs =
          get_parallel_tensor_attrs(pcg, t_relu_output);

      DynamicValueAttrs relu_key{
          dynamic_tensor_guid_t{t_relu_output},
          relu_output_attrs.shape,
          reduced_coord,
          bidict<ParallelTensorSpaceCoordinate, MachineSpaceCoordinate>{
              {reduced_coord, gpu0}},
          std::nullopt,
          DynamicTensorRole{FwbTensorType::FORWARD},
      };

      auto [relu_inst, relu_ready] = backing.backing.at(relu_key);

      Allocator cpu_allocator = ctx.get_current_device_allocator();

      GenericTensorAccessorR relu_gpu =
          dynamic_tensor_accessor_from_instance(relu_inst,
                                                relu_ready,
                                                relu_output_attrs.shape,
                                                Permissions::RO,
                                                ctx.get_current_processor())
              .get<GenericTensorAccessorR>();

      GenericTensorAccessorR relu_cpu =
          copy_tensor_accessor_r_to_cpu_if_necessary(relu_gpu, cpu_allocator);

      // expected: relu(relu(input @ weight.T) + relu(input @ weight.T))
      // = relu(4.0 + 4.0) = 8.0 for all elements
      float const *relu_ptr = relu_cpu.get_float_ptr();
      int output_num_elements = batch_size.int_from_positive_int() *
                                out_channels.int_from_positive_int();
      for (int i = 0; i < output_num_elements; i++) {
        INFO("index = ", i);
        CHECK_EQ(relu_ptr[i], 8.0f);
      }
    });
    result.wait();
  }
}
} // namespace test
