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
#include "pcg/mapped_parallel_computation_graph/mapped_parallel_computation_graph.h"
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

static bool did_loss_decrease(GenericTensorAccessorR const &first_epoch,
                              GenericTensorAccessorR const &last_epoch,
                              Allocator &allocator) {
  return tensor_accessor_all(
      compare_tensor_accessors_le(last_epoch, first_epoch, allocator));
}

MappedParallelComputationGraph
    make_test_mpcg_for_device_type(DeviceType device_type) {
  positive_int batch_size = 10_p;
  positive_int data_dim = 16_p;
  positive_int hidden_dim = 32_p;
  positive_int output_dim = 1_p;

  TensorShape output_tensor_shape = TensorShape{
      TensorDims{FFOrdered{batch_size, output_dim}}, DataType::FLOAT};

  TensorShape label_tensor_shape = TensorShape{
      TensorDims{FFOrdered{batch_size, output_dim}}, DataType::FLOAT};

  ParallelComputationGraph pcg = empty_parallel_computation_graph();

  TensorShape input_tensor_shape =
      TensorShape{TensorDims{FFOrdered{batch_size, data_dim}}, DataType::FLOAT};

  ParallelLayerAddedResult inputs_layer =
      pcg_add_input_layer(pcg, input_tensor_shape);
  parallel_tensor_guid_t t_input =
      require_only_key(inputs_layer.outputs, TensorSlotName::OUTPUT);

  ParallelLayerAddedResult inputs_layer_2 =
      pcg_add_input_layer(pcg, input_tensor_shape);
  parallel_tensor_guid_t t_input_2 =
      require_only_key(inputs_layer_2.outputs, TensorSlotName::OUTPUT);

  ElementBinaryAttrs add_attrs = ElementBinaryAttrs{
      OperatorType::EW_ADD,
      DataType::FLOAT,
      false,
      false,
  };

  ParallelLayerAddedResult add_operator_1 =
      add_parallel_layer(pcg,
                         make_layer_attrs(add_attrs),
                         {
                             {
                                 TensorSlotName::LHS_INPUT,
                                 t_input,
                             },
                             {
                                 TensorSlotName::RHS_INPUT,
                                 t_input_2,
                             },
                         },
                         /*weights=*/{});

  parallel_tensor_guid_t t_add_1 =
      require_only_key(add_operator_1.outputs, TensorSlotName::OUTPUT);

  positive_int replicate_degree = 2_p;
  ReplicateAttrs repl_attrs = ReplicateAttrs{replicate_degree};
  ParallelLayerAddedResult repl_operator_1 =
      add_parallel_layer(pcg,
                         make_layer_attrs(repl_attrs),
                         {
                             {
                                 TensorSlotName::INPUT,
                                 t_add_1,
                             },
                         },
                         /*weight=*/{});

  parallel_tensor_guid_t t_repl_1 =
      require_only_key(repl_operator_1.outputs, TensorSlotName::OUTPUT);

  ParallelLayerAddedResult relu_operator_1 =
      add_parallel_layer(pcg,
                         make_layer_attrs(make_relu_attrs()),
                         /*inputs=*/
                         {
                             {
                                 TensorSlotName::INPUT,
                                 t_repl_1,
                             },
                         },
                         /*weights=*/{});

  parallel_tensor_guid_t t_relu_1 =
      require_only_key(relu_operator_1.outputs, TensorSlotName::OUTPUT);

  MachineSpaceCoordinate cpu0{0_n, 0_n, device_type};
  MachineSpaceCoordinate cpu1{0_n, 1_n, device_type};

  ParallelTensorSpaceCoordinate tensor_coord0{
      /*sum_component=*/0_n,
      /*discard_copy_component=*/0_n,
      /*shard_component=*/FFOrdered{0_n}};
  ParallelTensorSpaceCoordinate tensor_coord1{
      /*sum_component=*/0_n,
      /*discard_copy_component=*/1_n,
      /*shard_component=*/FFOrdered{0_n}};

  MappedParallelComputationGraph mpcg =
      mapped_pcg_from_pcg_and_mapped_op_task_groups(
          /*pcg=*/pcg,
          /*mapped_op_task_groups=*/{
              {
                  inputs_layer.parallel_layer,
                  MappedOperatorTaskGroup{
                      {
                          {
                              cpu0,
                              OperatorAtomicTaskShardBinding{{
                                  {TensorSlotName::OUTPUT, tensor_coord0},
                              }},
                          },
                      },
                  },
              },
              {
                  inputs_layer_2.parallel_layer,
                  MappedOperatorTaskGroup{
                      {
                          {
                              cpu0,
                              OperatorAtomicTaskShardBinding{{
                                  {TensorSlotName::OUTPUT, tensor_coord0},
                              }},
                          },
                      },
                  },
              },
              {
                  add_operator_1.parallel_layer,
                  MappedOperatorTaskGroup{
                      {
                          {
                              cpu0,
                              OperatorAtomicTaskShardBinding{{
                                  {TensorSlotName::LHS_INPUT, tensor_coord0},
                                  {TensorSlotName::RHS_INPUT, tensor_coord0},
                                  {TensorSlotName::OUTPUT, tensor_coord0},
                              }},
                          },
                      },
                  },
              },
              {
                  repl_operator_1.parallel_layer,
                  MappedOperatorTaskGroup{
                      {
                          {
                              cpu0,
                              OperatorAtomicTaskShardBinding{{
                                  {TensorSlotName::OUTPUT, tensor_coord0},
                              }},
                          },
                          {
                              cpu1,
                              OperatorAtomicTaskShardBinding{{
                                  {TensorSlotName::OUTPUT, tensor_coord1},
                              }},
                          },
                      },
                  },
              },
              {
                  relu_operator_1.parallel_layer,
                  MappedOperatorTaskGroup{
                      {
                          {
                              cpu0,
                              OperatorAtomicTaskShardBinding{{
                                  {TensorSlotName::INPUT, tensor_coord0},
                                  {TensorSlotName::OUTPUT, tensor_coord0},
                              }},
                          },
                          {
                              cpu1,
                              OperatorAtomicTaskShardBinding{{
                                  {TensorSlotName::INPUT, tensor_coord1},
                                  {TensorSlotName::OUTPUT, tensor_coord1},
                              }},
                          },
                      },
                  },
              },
          });

  return mpcg;
}

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("RealmBackend e2e Training Replicate Op (CPU Model Parallelism)") {
    std::vector<char *> fake_args =
        make_fake_realm_args(/*num_cpus=*/2_p, /*num_gpus=*/0_n);
    int fake_argc = fake_args.size();
    char **fake_argv = fake_args.data();

    RealmManager manager = RealmManager{&fake_argc, &fake_argv};
    ControllerTaskResult result =
        manager.start_controller([](RealmContext &ctx) {
          Allocator allocator = ctx.get_current_device_allocator();

          MappedParallelComputationGraph mpcg =
              make_test_mpcg_for_device_type(DeviceType::CPU);

          std::unordered_map<DynamicValueAttrs, DynamicTensorAccessor>
              input_tensors;

          OptimizerAttrs optimizer_attrs = OptimizerAttrs{
              SGDOptimizerAttrs{
                  /*lr=*/0.001,
                  /*momentum=*/0.9,
                  /*nesterov=*/false,
                  /*weight_decay=*/0.001,
              },
          };

          DistributedFfHandle device_handle = create_distributed_ff_handle(
              ctx,
              /*workSpaceSize=*/1024 * 1024,
              /*allowTensorOpMathConversion=*/true);

          PCGInstance pcg_instance = create_pcg_instance(
              /*ctx=*/ctx,
              /*mpcg=*/mpcg,
              /*optimizer=*/optimizer_attrs,
              /*loss=*/std::nullopt,
              /*input_tensors=*/input_tensors,
              /*profiling_settings=*/ProfilingSettings{0, 0},
              /*device_handle=*/device_handle,
              /*iteration_config=*/FFIterationConfig{1_p});

          // begin training loop
          int num_epochs = 1;
          for (int i = 0; i < num_epochs; i++) {
            perform_all_passes_for_pcg_instance(
                /*instance=*/pcg_instance,
                /*profiling_settings=*/ProfilingSettings{0, 0},
                /*device_handle=*/device_handle,
                /*iteration_config=*/FFIterationConfig{1_p});
          }
        });
    result.wait();
  }
}

TEST_SUITE(FF_CUDA_TEST_SUITE) {
  TEST_CASE("RealmBackend e2e Training Replicate Op (GPU Model Parallelism)") {
    std::vector<char *> fake_args =
        make_fake_realm_args(/*num_cpus=*/1_p, /*num_gpus=*/2_n);
    int fake_argc = fake_args.size();
    char **fake_argv = fake_args.data();

    RealmManager manager = RealmManager{&fake_argc, &fake_argv};

    ControllerTaskResult result =
        manager.start_controller([](RealmContext &ctx) {
          Allocator allocator = ctx.get_current_device_allocator();

          MappedParallelComputationGraph mpcg =
              make_test_mpcg_for_device_type(DeviceType::GPU);

          OptimizerAttrs optimizer_attrs = OptimizerAttrs{
              SGDOptimizerAttrs{
                  /*lr=*/0.001,
                  /*momentum=*/0.9,
                  /*nesterov=*/false,
                  /*weight_decay=*/0.001,
              },
          };

          std::unordered_map<DynamicValueAttrs, DynamicTensorAccessor>
              input_tensors;

          DistributedFfHandle device_handle = create_distributed_ff_handle(
              ctx,
              /*workSpaceSize=*/1024 * 1024,
              /*allowTensorOpMathConversion=*/true);

          PCGInstance pcg_instance = create_pcg_instance(
              /*ctx=*/ctx,
              /*mpcg=*/mpcg,
              /*optimizer=*/optimizer_attrs,
              /*loss=*/std::nullopt,
              /*input_tensors=*/input_tensors,
              /*profiling_settings=*/ProfilingSettings{0, 0},
              /*device_handle=*/device_handle,
              /*iteration_config=*/FFIterationConfig{1_p});

          // begin training loop
          int num_epochs = 1;
          for (int i = 0; i < num_epochs; i++) {
            perform_all_passes_for_pcg_instance(
                /*instance=*/pcg_instance,
                /*profiling_settings=*/ProfilingSettings{0, 0},
                /*device_handle=*/device_handle,
                /*iteration_config=*/FFIterationConfig{1_p});
          }
        });
    result.wait();
  }
}
} // namespace test
