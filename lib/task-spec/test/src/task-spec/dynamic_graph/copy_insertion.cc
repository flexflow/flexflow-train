#include "task-spec/dynamic_graph/copy_insertion.h"
#include "op-attrs/tensor_slot_name.dtg.h"
#include "pcg/mapped_parallel_computation_graph/mapped_operator_task_group.h"
#include "task-spec/dynamic_graph/dynamic_value_attrs.dtg.h"
#include "test/utils/doctest/fmt/unordered_set.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("perform_copy_insertion_for_invocation") {
    auto mk_machine_coord =
        [](nonnegative_int node_idx,
           nonnegative_int device_idx) -> MachineSpaceCoordinate {
      return MachineSpaceCoordinate{
          /*node_idx=*/node_idx,
          /*device_idx=*/device_idx,
          /*device_type=*/DeviceType::GPU,
      };
    };

    auto mk_pt_coord =
        [](nonnegative_int idx1,
           nonnegative_int idx2,
           nonnegative_int idx3,
           nonnegative_int idx4) -> ParallelTensorSpaceCoordinate {
      return ParallelTensorSpaceCoordinate{
          /*sum_component=*/idx1,
          /*discard_copy_component=*/idx2,
          /*shard_components=*/
          FFOrdered{
              idx3,
              idx4,
          },
      };
    };

    auto mk_input_shard_binding = [&](ParallelTensorSpaceCoordinate const &c)
        -> OperatorAtomicTaskShardBinding {
      return OperatorAtomicTaskShardBinding{
          /*tensor_coords=*/{
              {
                  TensorSlotName::OUTPUT,
                  c,
              },
          },
      };
    };

    auto mk_shard_binding = [&](ParallelTensorSpaceCoordinate const &c1,
                                ParallelTensorSpaceCoordinate const &c2,
                                ParallelTensorSpaceCoordinate const &c3,
                                ParallelTensorSpaceCoordinate const &c4)
        -> OperatorAtomicTaskShardBinding {
      return OperatorAtomicTaskShardBinding{
          /*tensor_coords=*/{
              {
                  TensorSlotName::INPUT,
                  c1,
              },
              {
                  TensorSlotName::WEIGHT,
                  c2,
              },
              {
                  TensorSlotName::OUTPUT_1,
                  c3,
              },
              {
                  TensorSlotName::OUTPUT_2,
                  c4,
              },
          },
      };
    };

    MachineSpaceCoordinate mc1 = mk_machine_coord(0_n, 0_n);
    MachineSpaceCoordinate mc2 = mk_machine_coord(1_n, 0_n);
    MachineSpaceCoordinate mc3 = mk_machine_coord(2_n, 0_n);
    MachineSpaceCoordinate mc4 = mk_machine_coord(3_n, 0_n);

    ParallelTensorSpaceCoordinate mc1_input_coord =
        mk_pt_coord(0_n, 0_n, 0_n, 0_n);
    ParallelTensorSpaceCoordinate mc1_weight_coord =
        mk_pt_coord(0_n, 1_n, 2_n, 0_n);
    ParallelTensorSpaceCoordinate mc1_output_1_coord =
        mk_pt_coord(1_n, 0_n, 0_n, 1_n);
    ParallelTensorSpaceCoordinate mc1_output_2_coord =
        mk_pt_coord(3_n, 0_n, 0_n, 0_n);

    ParallelTensorSpaceCoordinate mc2_input_coord =
        mk_pt_coord(0_n, 1_n, 0_n, 0_n);
    ParallelTensorSpaceCoordinate mc2_weight_coord =
        mk_pt_coord(0_n, 4_n, 2_n, 0_n);
    ParallelTensorSpaceCoordinate mc2_output_1_coord =
        mk_pt_coord(1_n, 2_n, 0_n, 1_n);
    ParallelTensorSpaceCoordinate mc2_output_2_coord =
        mk_pt_coord(0_n, 0_n, 0_n, 0_n);

    MappedOperatorTaskGroup input_mapping_same = MappedOperatorTaskGroup{
        bidict<MachineSpaceCoordinate, OperatorAtomicTaskShardBinding>{
            {
                mc1,
                mk_input_shard_binding(mc1_input_coord),
            },
            {
                mc2,
                mk_input_shard_binding(mc2_input_coord),
            },
        },
    };
    MappedOperatorTaskGroup input_mapping_copy1 = MappedOperatorTaskGroup{
        bidict<MachineSpaceCoordinate, OperatorAtomicTaskShardBinding>{
            {
                mc1,
                mk_input_shard_binding(mc1_input_coord),
            },
            {
                mc3,
                mk_input_shard_binding(mc2_input_coord),
            },
        },
    };
    MappedOperatorTaskGroup input_mapping_copy1_diff_vs_use =
        MappedOperatorTaskGroup{
            bidict<MachineSpaceCoordinate, OperatorAtomicTaskShardBinding>{
                {
                    mc3,
                    mk_input_shard_binding(mc2_input_coord),
                },
            },
        };
    MappedOperatorTaskGroup input_mapping_copy2 = MappedOperatorTaskGroup{
        bidict<MachineSpaceCoordinate, OperatorAtomicTaskShardBinding>{
            {
                mc3,
                mk_input_shard_binding(mc1_input_coord),
            },
            {
                mc4,
                mk_input_shard_binding(mc2_input_coord),
            },
        },
    };

    MappedOperatorTaskGroup weight_mapping_same = MappedOperatorTaskGroup{
        bidict<MachineSpaceCoordinate, OperatorAtomicTaskShardBinding>{
            {
                mc1,
                mk_input_shard_binding(mc1_weight_coord),
            },
            {
                mc2,
                mk_input_shard_binding(mc2_weight_coord),
            },
        },
    };
    MappedOperatorTaskGroup weight_mapping_copy2 = MappedOperatorTaskGroup{
        bidict<MachineSpaceCoordinate, OperatorAtomicTaskShardBinding>{
            {
                mc4,
                mk_input_shard_binding(mc1_weight_coord),
            },
            {
                mc3,
                mk_input_shard_binding(mc2_weight_coord),
            },
        },
    };

    MappedOperatorTaskGroup invocation_mapping = MappedOperatorTaskGroup{
        bidict<MachineSpaceCoordinate, OperatorAtomicTaskShardBinding>{
            {
                mc1,
                mk_shard_binding(mc1_input_coord,
                                 mc1_weight_coord,
                                 mc1_output_1_coord,
                                 mc1_output_2_coord),
            },
            {
                mc2,
                mk_shard_binding(mc2_input_coord,
                                 mc2_weight_coord,
                                 mc2_output_1_coord,
                                 mc2_output_2_coord),
            },
        },
    };

    MappedOperatorTaskGroup invocation_mapping_diff_vs_copy1 =
        MappedOperatorTaskGroup{
            bidict<MachineSpaceCoordinate, OperatorAtomicTaskShardBinding>{
                {
                    mc2,
                    mk_shard_binding(mc2_input_coord,
                                     mc2_weight_coord,
                                     mc2_output_1_coord,
                                     mc2_output_2_coord),
                },
            },
        };
    auto mk_slot = [](TensorSlotName const &slot_name) -> DynamicTensorSlot {
      return DynamicTensorSlot{
          /*slot_name=*/slot_name,
          /*slot_tensor_role=*/std::nullopt,
      };
    };

    auto mk_value = [&](size_t src_node_id,
                        TensorSlotName src_slot_name,
                        MappedOperatorTaskGroup const &mapping,
                        std::optional<TensorSlotName> const &use_slot_name)
        -> DynamicValueAttrs {
      return DynamicValueAttrs{
          /*tensor_guid=*/dynamic_tensor_guid_t{parallel_tensor_guid_t{
              KwargDataflowOutput<TensorSlotName>{
                  Node{src_node_id},
                  src_slot_name,
              },
          }},
          /*parallel_tensor_shape=*/std::nullopt,
          /*shard_coord=*/std::nullopt,
          /*mapping=*/
          transform(use_slot_name,
                    [&](TensorSlotName s) {
                      return get_tensor_bindings_for_slot_name(mapping, s);
                    }),
          /*accessor=*/std::nullopt,
          /*role=*/std::nullopt,
      };
    };

    size_t invocation1_id = 20;

    DynamicValueAttrs graph_input1 =
        mk_value(0, TensorSlotName::OUTPUT, invocation_mapping, std::nullopt);
    DynamicValueAttrs graph_input1_use = mk_value(
        0, TensorSlotName::OUTPUT, invocation_mapping, TensorSlotName::INPUT);
    DynamicValueAttrs graph_input1_use_diff_vs_copy1 =
        mk_value(0,
                 TensorSlotName::OUTPUT,
                 invocation_mapping_diff_vs_copy1,
                 TensorSlotName::INPUT);
    DynamicValueAttrs graph_input2 =
        mk_value(1, TensorSlotName::OUTPUT, invocation_mapping, std::nullopt);
    DynamicValueAttrs graph_input2_use = mk_value(
        1, TensorSlotName::OUTPUT, invocation_mapping, TensorSlotName::WEIGHT);
    DynamicValueAttrs invocation1_output1 = mk_value(invocation1_id,
                                                     TensorSlotName::OUTPUT_1,
                                                     invocation_mapping,
                                                     std::nullopt);
    DynamicValueAttrs invocation1_output1_src =
        mk_value(invocation1_id,
                 TensorSlotName::OUTPUT_1,
                 invocation_mapping,
                 TensorSlotName::OUTPUT_1);
    DynamicValueAttrs invocation1_output2 = mk_value(invocation1_id,
                                                     TensorSlotName::OUTPUT_2,
                                                     invocation_mapping,
                                                     std::nullopt);
    DynamicValueAttrs invocation1_output2_src =
        mk_value(invocation1_id,
                 TensorSlotName::OUTPUT_2,
                 invocation_mapping,
                 TensorSlotName::OUTPUT_2);

    DynamicValueAttrs graph_input1_src_same = mk_value(
        0, TensorSlotName::OUTPUT, input_mapping_same, TensorSlotName::OUTPUT);
    DynamicValueAttrs graph_input1_src_copy1 = mk_value(
        0, TensorSlotName::OUTPUT, input_mapping_copy1, TensorSlotName::OUTPUT);
    DynamicValueAttrs graph_input1_src_copy1_diff_vs_use =
        mk_value(0,
                 TensorSlotName::OUTPUT,
                 input_mapping_copy1_diff_vs_use,
                 TensorSlotName::OUTPUT);
    DynamicValueAttrs graph_input1_src_copy2 = mk_value(
        0, TensorSlotName::OUTPUT, input_mapping_copy2, TensorSlotName::OUTPUT);
    DynamicValueAttrs graph_input2_src_same = mk_value(
        1, TensorSlotName::OUTPUT, weight_mapping_same, TensorSlotName::OUTPUT);
    DynamicValueAttrs graph_input2_src_copy2 = mk_value(1,
                                                        TensorSlotName::OUTPUT,
                                                        weight_mapping_copy2,
                                                        TensorSlotName::OUTPUT);

    DynamicNodeInvocation input = DynamicNodeInvocation{
        /*inputs=*/{
            {
                mk_slot(TensorSlotName::INPUT),
                graph_input1,
            },
            {
                mk_slot(TensorSlotName::WEIGHT),
                graph_input2,
            },
        },
        /*node_attrs=*/
        DynamicNodeAttrs{
            /*task_type=*/std::nullopt,
            /*device_coord=*/std::nullopt,
            /*mapping=*/invocation_mapping,
            /*op_attrs=*/std::nullopt,
            /*layer_guid=*/
            dynamic_layer_guid_t{parallel_layer_guid_t{Node{20}}},
            /*per_device_op_state=*/std::nullopt,
        },
        /*outputs=*/
        {
            {
                mk_slot(TensorSlotName::OUTPUT_1),
                invocation1_output1,
            },
            {
                mk_slot(TensorSlotName::OUTPUT_2),
                invocation1_output2,
            },
        },
    };

    DynamicNodeInvocation mapped = DynamicNodeInvocation{
        /*inputs=*/{
            {
                mk_slot(TensorSlotName::INPUT),
                graph_input1_use,
            },
            {
                mk_slot(TensorSlotName::WEIGHT),
                graph_input2_use,
            },
        },
        /*node_attrs=*/
        DynamicNodeAttrs{
            /*task_type=*/std::nullopt,
            /*device_coord=*/std::nullopt,
            /*mapping=*/invocation_mapping,
            /*op_attrs=*/std::nullopt,
            /*layer_guid=*/
            dynamic_layer_guid_t{parallel_layer_guid_t{Node{20}}},
            /*per_device_op_state=*/std::nullopt,
        },
        /*outputs=*/
        {
            {
                mk_slot(TensorSlotName::OUTPUT_1),
                invocation1_output1_src,
            },
            {
                mk_slot(TensorSlotName::OUTPUT_2),
                invocation1_output2_src,
            },
        },
    };

    auto mk_copy = [&](DynamicValueAttrs const &src,
                       DynamicValueAttrs const &dst) {
      return DynamicNodeInvocation{
          /*inputs=*/{{mk_slot(TensorSlotName::INPUT), src}},
          /*node_attrs=*/
          DynamicNodeAttrs{
              /*task_type=*/std::nullopt,
              /*device_coord=*/std::nullopt,
              /*mapping=*/std::nullopt,
              /*op_attrs*/ TrainingOperationAttrs{CopyAttrs{}},
              /*layer_guid=*/dynamic_layer_guid_t{dynamic_copy_layer_guid_t{}},
              /*per_device_op_state=*/std::nullopt,
          },
          /*outputs=*/{{mk_slot(TensorSlotName::OUTPUT), dst}},
      };
    };

    SUBCASE("same mapping, no copies") {
      std::unordered_map<DynamicValueAttrs, DynamicValueAttrs> sources_same{
          {graph_input1, graph_input1_src_same},
          {graph_input2, graph_input2_src_same}};

      std::unordered_set<DynamicNodeInvocation> result =
          perform_copy_insertion_for_invocation(input, sources_same);

      std::unordered_set<DynamicNodeInvocation> correct = {mapped};

      CHECK(result.size() == correct.size());
      CHECK(result == correct);
    }

    SUBCASE("copy one tensor, one point") {
      std::unordered_map<DynamicValueAttrs, DynamicValueAttrs> sources_copy1{
          {graph_input1, graph_input1_src_copy1},
          {graph_input2, graph_input2_src_same}};

      std::unordered_set<DynamicNodeInvocation> result =
          perform_copy_insertion_for_invocation(input, sources_copy1);

      std::unordered_set<DynamicNodeInvocation> correct = {
          mapped,
          mk_copy(graph_input1_src_copy1_diff_vs_use,
                  graph_input1_use_diff_vs_copy1),
      };

      CHECK(result.size() == correct.size());
      CHECK(result == correct);
    }

    SUBCASE("copy two tensors, two points") {
      std::unordered_map<DynamicValueAttrs, DynamicValueAttrs> sources_copy2{
          {graph_input1, graph_input1_src_copy2},
          {graph_input2, graph_input2_src_copy2}};

      std::unordered_set<DynamicNodeInvocation> result =
          perform_copy_insertion_for_invocation(input, sources_copy2);

      std::unordered_set<DynamicNodeInvocation> correct = {
          mapped,
          mk_copy(graph_input1_src_copy2, graph_input1_use),
          mk_copy(graph_input2_src_copy2, graph_input2_use),
      };

      CHECK(result.size() == correct.size());
      CHECK(result == correct);
    }
  }
}
