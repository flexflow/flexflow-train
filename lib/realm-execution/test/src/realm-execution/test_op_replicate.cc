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
  TEST_CASE("RealmBackend e2e Training Replicate Op (GPU Model Parallelism)") {
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
          positive_int hidden_dim = 32_p;
          positive_int output_dim = 1_p;

          // 10,2
          TensorShape output_tensor_shape = TensorShape{
              TensorDims{FFOrdered{batch_size, output_dim}}, DataType::FLOAT};

          // 10,2
          TensorShape label_tensor_shape = TensorShape{
              TensorDims{FFOrdered{batch_size, output_dim}}, DataType::FLOAT};

          GenericTensorAccessorW label_tensor =
              allocator.allocate_tensor(label_tensor_shape);

          // construct computation graph
          ParallelComputationGraph pcg = empty_parallel_computation_graph();

          // input tensor
          // 10, 16
          TensorShape input_tensor_shape = TensorShape{
              TensorDims{FFOrdered{batch_size, data_dim}}, DataType::FLOAT};

          // parallel layer -> input tensor
          ParallelLayerAddedResult inputs_layer =
              pcg_add_input_layer(pcg, input_tensor_shape);
          parallel_tensor_guid_t t_input =
              require_only_key(inputs_layer.outputs, TensorSlotName::OUTPUT);

          // parallel layer -> input tensor 2
          ParallelLayerAddedResult inputs_layer_2 =
              pcg_add_input_layer(pcg, input_tensor_shape);
          parallel_tensor_guid_t t_input_2 =
              require_only_key(inputs_layer_2.outputs, TensorSlotName::OUTPUT);

          // binary  ADD attribute
          ElementBinaryAttrs add_attrs = ElementBinaryAttrs{
              OperatorType::EW_ADD,
              DataType::FLOAT,
              false,
              false,
          };

          // parallel layer -> perform add
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
                                 {/* weight */});

          parallel_tensor_guid_t t_add_1 =
              require_only_key(add_operator_1.outputs, TensorSlotName::OUTPUT);

          // parallel layer -> perform replicate
          const positive_int replicate_degree = 2_p;
          ReplicateAttrs repl_attrs = ReplicateAttrs(replicate_degree);
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
          // output of replicate layer
          parallel_tensor_guid_t t_repl_1 =
              require_only_key(repl_operator_1.outputs, TensorSlotName::OUTPUT);

          // parallel layer -> perform  RelU
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
          // output of relu layer
          parallel_tensor_guid_t t_relu_1 =
              require_only_key(relu_operator_1.outputs, TensorSlotName::OUTPUT);

          // machine
          MachineSpaceCoordinate gpu0{0_n, 0_n, DeviceType::GPU};
          MachineSpaceCoordinate gpu1{0_n, 1_n, DeviceType::GPU};
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
                  {inputs_layer_2.parallel_layer,
                   MappedOperatorTaskGroup{
                       {{gpu0,
                         OperatorAtomicTaskShardBinding{
                             {{TensorSlotName::OUTPUT, tensor_coord0}}}}}}},
                  {add_operator_1.parallel_layer,
                   MappedOperatorTaskGroup{
                       {{gpu0,
                         OperatorAtomicTaskShardBinding{{
                             {TensorSlotName::LHS_INPUT, tensor_coord0},
                             {TensorSlotName::RHS_INPUT, tensor_coord0},
                             {TensorSlotName::OUTPUT, tensor_coord0},
                         }}}}}},
                  {repl_operator_1.parallel_layer,
                   MappedOperatorTaskGroup{
                       {{gpu0,
                         OperatorAtomicTaskShardBinding{{
                             {TensorSlotName::OUTPUT, tensor_coord0},
                         }}},
                        {gpu1,
                         OperatorAtomicTaskShardBinding{{
                             {TensorSlotName::OUTPUT, tensor_coord1},
                         }}}}}},
                  {relu_operator_1.parallel_layer,
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
              },
          };

          MappedOperatorTaskGroup loss_mapping{
              {{gpu0,
                OperatorAtomicTaskShardBinding{{
                    {TensorSlotName::INPUT, tensor_coord0},
                    {TensorSlotName::LOGIT, tensor_coord0},
                }}}}};

          // instantiate computation graph
          LossAttrs loss_attrs = LossAttrs{
              NonconfigurableLossAttrs{LossFunction::CATEGORICAL_CROSSENTROPY}};
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

          PCGInstance pcg_instance = create_pcg_instance(
              /*ctx=*/ctx,
              /*mpcg=*/mpcg,
              /*optimizer=*/optimizer_attrs,
              /*loss=*/std::nullopt,
              /*input_tensors=*/input_tensors,
              /*profiling_settings=*/ProfilingSettings{0_n, 1_p},
              /*device_handle=*/device_handle,
              /*iteration_config=*/FFIterationConfig{1_p});

          // begin training loop
          int num_epochs = 1;
          for (int i = 0; i < num_epochs; i++) {
            perform_all_passes_for_pcg_instance(
                /*instance=*/pcg_instance,
                /*profiling_settings=*/ProfilingSettings{0_n, 1_p},
                /*device_handle=*/device_handle,
                /*iteration_config=*/FFIterationConfig{1_p});
          }
        });
    result.wait();
  }
}
} // namespace test
