#include <doctest/doctest.h>
#include "task-spec/dynamic_graph/make_dynamic_open_dataflow_graph_from_mapped_pcg.h"
#include "utils/containers/require_only_key.h"
#include "op-attrs/ops/element_unary.h"
#include "pcg/mapped_parallel_computation_graph/mapped_parallel_computation_graph.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("make_dynamic_node_invocation_from_mapped") {
    SUBCASE("Replicate") {
      MachineSpaceCoordinate gpu0 = MachineSpaceCoordinate{0_n, 0_n, DeviceType::GPU};
      MachineSpaceCoordinate gpu1 = MachineSpaceCoordinate{0_n, 1_n, DeviceType::GPU};

      ParallelTensorSpaceCoordinate tensor_coord0 = ParallelTensorSpaceCoordinate{
        /*sum_component=*/0_n,
        /*discard_copy_component=*/0_n,
        /*shard_component=*/FFOrdered{0_n},
      };

      ParallelTensorSpaceCoordinate tensor_coord1 = ParallelTensorSpaceCoordinate{
        /*sum_component=*/0_n,
        /*discard_copy_component=*/1_n,
        /*shard_component=*/FFOrdered{0_n},
      };

      MappedOperatorTaskGroup mapping = MappedOperatorTaskGroup{
        {
          {
            gpu0,
            OperatorAtomicTaskShardBinding{{
              {TensorSlotName::OUTPUT, tensor_coord0},
            }},
          },
          {
            gpu1,
            OperatorAtomicTaskShardBinding{{
               {TensorSlotName::OUTPUT, tensor_coord1},
            }},
          },
        },
      };

      ParallelTensorShape input_shape = ParallelTensorShape{
        /*dims=*/ParallelTensorDims{
          /*shard_dims=*/FFOrdered<ShardParallelDim>{
            ShardParallelDim{8_p, 2_p},
            ShardParallelDim{5_p, 1_p},
          },
          /*replica_dims=*/ReplicaParallelDimSet{
            SumDegree{1_p},
            DiscardCopyDegree{1_p},
          },
        },
        /*data_type=*/DataType::FLOAT,
      };

      ParallelTensorShape output_shape = [&] {
        ParallelTensorShape shape = input_shape;
        shape.dims.replica_dims.discard_copy_degree = DiscardCopyDegree{2_p};
        return shape;
      }();

      PCGOperatorAttrs op_attrs = PCGOperatorAttrs{
        ReplicateAttrs{
          2_p,
        },
      };

      parallel_layer_guid_t layer_guid = parallel_layer_guid_t{Node{0}};
      parallel_tensor_guid_t input_tensor_guid = parallel_tensor_guid_t{
        KwargDataflowOutput{
          Node{5},
          TensorSlotName::OUTPUT,
        },
      };
      parallel_tensor_guid_t output_tensor_guid = parallel_tensor_guid_t{
        KwargDataflowOutput{
          Node{0},
          TensorSlotName::OUTPUT,
        },
      };

      MappedParallelLayerInvocationInfo input = MappedParallelLayerInvocationInfo{
        /*incoming=*/{
          {
            TensorSlotName::INPUT,
            ParallelTensorInfo{
              /*guid=*/input_tensor_guid,
              /*attrs=*/ParallelTensorAttrs{
                /*shape=*/input_shape,
                /*create_grad=*/CreateGrad::YES,
              },
            },
          },
        },
        /*layer_info=*/MappedParallelLayerInfo{
          /*guid=*/layer_guid,
          /*attrs=*/ParallelLayerAttrs{
            /*op_attrs=*/op_attrs,
            /*name=*/std::nullopt,
          },
          /*mapping=*/mapping,
        },
        /*outgoing=*/{
          {
            TensorSlotName::OUTPUT,
            ParallelTensorInfo{
              /*guid=*/output_tensor_guid,
              /*attrs=*/ParallelTensorAttrs{
                /*shape=*/output_shape,
                /*create_grad=*/CreateGrad::YES,
              },
            },
          },
        },
      };

      DynamicNodeInvocation result = make_dynamic_node_invocation_from_mapped(input);

      DynamicNodeInvocation correct = DynamicNodeInvocation{
        /*inputs=*/{
          {
            DynamicTensorSlot{
              TensorSlotName::INPUT,
              /*slot_tensor_role=*/std::nullopt,
              /*task_shard=*/std::nullopt,
            },
            DynamicValueAttrs{
              /*tensor_guid=*/dynamic_tensor_guid_t{input_tensor_guid},
              /*parallel_tensor_shape=*/input_shape,
              /*shard_coord=*/std::nullopt,
              /*mapping=*/std::nullopt,
              /*accessor=*/std::nullopt,
              /*role=*/std::nullopt,
            },
          },
        },
        /*node_attrs=*/DynamicNodeAttrs{
          /*task_type=*/std::nullopt,
          /*device_coord=*/std::nullopt,
          /*mapping=*/mapping,
          /*op_attrs=*/TrainingOperationAttrs{op_attrs},
          /*layer_guid=*/dynamic_layer_guid_t{layer_guid},
          /*per_device_op_state=*/std::nullopt,
        },
        /*outputs=*/{
          {
            DynamicTensorSlot{
              TensorSlotName::OUTPUT,
              /*slot_tensor_role=*/std::nullopt,
              /*task_shard=*/std::nullopt,
            },
            DynamicValueAttrs{
              /*tensor_guid=*/dynamic_tensor_guid_t{output_tensor_guid},
              /*parallel_tensor_shape=*/output_shape,
              /*shard_coord=*/std::nullopt,
              /*mapping=*/std::nullopt,
              /*accessor=*/std::nullopt,
              /*role=*/std::nullopt,
            },
          },
        }
      };

      CHECK(result == correct);
    }

    // SUBCASE("standard op") {
    //
    // }
  }

  // TEST_CASE("make_dynamic_open_dataflow_graph_from_mapped_pcg") {
  //   positive_int batch_size = 10_p;
  //   positive_int data_dim = 16_p;
  //   positive_int hidden_dim = 32_p;
  //   positive_int output_dim = 1_p;

  //   auto make_layer_attrs = [](auto const &op_attrs) -> ParallelLayerAttrs {
  //     return ParallelLayerAttrs{
  //         /*op_attrs=*/PCGOperatorAttrs{op_attrs},
  //         /*name=*/std::nullopt,
  //     };
  //   };


  //   TensorShape output_tensor_shape = TensorShape{
  //       TensorDims{FFOrdered{batch_size, output_dim}}, DataType::FLOAT};

  //   TensorShape label_tensor_shape = TensorShape{
  //       TensorDims{FFOrdered{batch_size, output_dim}}, DataType::FLOAT};

  //   ParallelComputationGraph pcg = empty_parallel_computation_graph();

  //   TensorShape input_tensor_shape = TensorShape{
  //       TensorDims{FFOrdered{batch_size, data_dim}}, DataType::FLOAT};

  //   ParallelLayerAddedResult inputs_layer =
  //       pcg_add_input_layer(pcg, input_tensor_shape);
  //   parallel_tensor_guid_t t_input =
  //       require_only_key(inputs_layer.outputs, TensorSlotName::OUTPUT);

  //   ParallelLayerAddedResult inputs_layer_2 =
  //       pcg_add_input_layer(pcg, input_tensor_shape);
  //   parallel_tensor_guid_t t_input_2 =
  //       require_only_key(inputs_layer_2.outputs, TensorSlotName::OUTPUT);

  //   ElementBinaryAttrs add_attrs = ElementBinaryAttrs{
  //       OperatorType::EW_ADD,
  //       DataType::FLOAT,
  //       false,
  //       false,
  //   };

  //   ParallelLayerAddedResult add_operator_1 =
  //       add_parallel_layer(pcg,
  //                          make_layer_attrs(add_attrs),
  //                          {
  //                              {
  //                                  TensorSlotName::LHS_INPUT,
  //                                  t_input,
  //                              },
  //                              {
  //                                  TensorSlotName::RHS_INPUT,
  //                                  t_input_2,
  //                              },
  //                          },
  //                          /*weights=*/{});

  //   parallel_tensor_guid_t t_add_1 =
  //       require_only_key(add_operator_1.outputs, TensorSlotName::OUTPUT);

  //   positive_int replicate_degree = 2_p;
  //   ReplicateAttrs repl_attrs = ReplicateAttrs{replicate_degree};
  //   ParallelLayerAddedResult repl_operator_1 =
  //       add_parallel_layer(pcg,
  //                          make_layer_attrs(repl_attrs),
  //                          {
  //                              {
  //                                  TensorSlotName::INPUT,
  //                                  t_add_1,
  //                              },
  //                          },
  //                          /*weight=*/{});

  //   parallel_tensor_guid_t t_repl_1 =
  //       require_only_key(repl_operator_1.outputs, TensorSlotName::OUTPUT);

  //   ParallelLayerAddedResult relu_operator_1 =
  //       add_parallel_layer(pcg,
  //                          make_layer_attrs(make_relu_attrs()),
  //                          /*inputs=*/
  //                          {
  //                              {
  //                                  TensorSlotName::INPUT,
  //                                  t_repl_1,
  //                              },
  //                          },
  //                          /*weights=*/{});

  //   parallel_tensor_guid_t t_relu_1 =
  //       require_only_key(relu_operator_1.outputs, TensorSlotName::OUTPUT);

  //   MachineSpaceCoordinate gpu0{0_n, 0_n, DeviceType::GPU};
  //   MachineSpaceCoordinate gpu1{0_n, 1_n, DeviceType::GPU};

  //   ParallelTensorSpaceCoordinate tensor_coord0{
  //       /*sum_component=*/0_n,
  //       /*discard_copy_component=*/0_n,
  //       /*shard_component=*/FFOrdered{0_n}};
  //   ParallelTensorSpaceCoordinate tensor_coord1{
  //       /*sum_component=*/0_n,
  //       /*discard_copy_component=*/1_n,
  //       /*shard_component=*/FFOrdered{0_n}};

  //   MappedOperatorTaskGroup input_1_mapping = MappedOperatorTaskGroup{
  //     {
  //       {
  //         gpu0,
  //         OperatorAtomicTaskShardBinding{{
  //           {TensorSlotName::OUTPUT, tensor_coord0},
  //         }},
  //       },
  //     },
  //   };

  //   MappedOperatorTaskGroup input_2_mapping = MappedOperatorTaskGroup{
  //     {
  //       {
  //         gpu0,
  //         OperatorAtomicTaskShardBinding{{
  //           {TensorSlotName::OUTPUT, tensor_coord0},
  //         }},
  //       },
  //     },
  //   };

  //   MappedOperatorTaskGroup add_operator_1_mapping = MappedOperatorTaskGroup{
  //     {
  //       {
  //         gpu0,
  //         OperatorAtomicTaskShardBinding{{
  //             {TensorSlotName::LHS_INPUT, tensor_coord0},
  //             {TensorSlotName::RHS_INPUT, tensor_coord0},
  //             {TensorSlotName::OUTPUT, tensor_coord0},
  //         }},
  //       },
  //     },
  //   };

  //   MappedOperatorTaskGroup repl_operator_1_mapping = MappedOperatorTaskGroup{
  //     {
  //       {
  //         gpu0,
  //         OperatorAtomicTaskShardBinding{{
  //           {TensorSlotName::OUTPUT, tensor_coord0},
  //         }},
  //       },
  //       {
  //         gpu1,
  //         OperatorAtomicTaskShardBinding{{
  //            {TensorSlotName::OUTPUT, tensor_coord1},
  //         }},
  //       },
  //     },
  //   };

  //   MappedOperatorTaskGroup relu_operator_1_mapping = MappedOperatorTaskGroup{
  //     {
  //       {
  //         gpu0,
  //         OperatorAtomicTaskShardBinding{{
  //           {TensorSlotName::INPUT, tensor_coord0},
  //           {TensorSlotName::OUTPUT, tensor_coord0},
  //         }},
  //       },
  //       {
  //         gpu1,
  //         OperatorAtomicTaskShardBinding{{
  //           {TensorSlotName::INPUT, tensor_coord1},
  //           {TensorSlotName::OUTPUT, tensor_coord1},
  //         }},
  //       },
  //     },
  //   };

  //   MappedParallelComputationGraph mpcg = mapped_pcg_from_pcg_and_mapped_op_task_groups(
  //       /*pcg=*/pcg,
  //       /*mapped_op_task_groups=*/{
  //         {
  //           inputs_layer.parallel_layer,
  //           input_1_mapping,
  //         },
  //         {
  //           inputs_layer_2.parallel_layer,
  //           input_2_mapping,
  //         },
  //         {
  //           add_operator_1.parallel_layer,
  //           add_operator_1_mapping,
  //         },
  //         {
  //           repl_operator_1.parallel_layer,
  //           repl_operator_1_mapping,
  //         },
  //         {
  //           relu_operator_1.parallel_layer,
  //           relu_operator_1_mapping,
  //         },
  //       });


  //   DynamicOpenDataflowGraph result = make_dynamic_open_dataflow_graph_from_mapped_pcg(mpcg);

  //   DynamicNodeInvocation input_1_invocation = DynamicNodeInvocation{
  //     DynamicNodeAttrs{
  //       /*task_type=*/std::nullopt,
  //       /*device_coord=*/std::nullopt,
  //       /*mapping=*/input_1_mapping,
  //       /*op_attrs=*/TrainingOperationAttrs{
  //       /*pcg_layer_guid=*/
  //       /*per_device_op_state=*/std::nullopt,
  //     },
  //   };

  //   DynamicNodeInvocation input_2_invocation =
  //   DynamicNodeInvocation add_operator_1_invocation =
  //   DynamicNodeInvocation repl_operator_1_invocation =
  //   DynamicNodeInvocation relu_operator_1_invocation =

  //   DynamicOpenDataflowGraph correct = dynamic_open_dataflow_graph_from_invocation_set(
  //     /*invocations=*/{

  //     },
  //   };
  // }
}
