#include <doctest/doctest.h>
#include "task-spec/dynamic_graph/shard_expansion.h"
#include "test/utils/doctest/fmt/unordered_set.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("perform_shard_expansion_for_invocation") {
    auto mk_machine_coord = [](nonnegative_int node_idx, nonnegative_int device_idx) 
      -> MachineSpaceCoordinate
    {
      return MachineSpaceCoordinate{
        /*node_idx=*/node_idx,
        /*device_idx=*/device_idx,
        /*device_type=*/DeviceType::GPU,
      };
    };

    auto mk_pt_coord = [](nonnegative_int idx1, nonnegative_int idx2, nonnegative_int idx3, nonnegative_int idx4) 
      -> ParallelTensorSpaceCoordinate
    {
      return ParallelTensorSpaceCoordinate{
        /*sum_component=*/idx1,
        /*discard_copy_component=*/idx2,
        /*shard_components=*/FFOrdered{
          idx3,
          idx4,
        },
      };
    };

    auto mk_shard_binding = [&](ParallelTensorSpaceCoordinate const &c1,
                                ParallelTensorSpaceCoordinate const &c2, 
                                ParallelTensorSpaceCoordinate const &c3,
                                ParallelTensorSpaceCoordinate const &c4) 
      -> OperatorAtomicTaskShardBinding
    {
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
    MachineSpaceCoordinate mc2 = mk_machine_coord(2_n, 0_n);

    ParallelTensorSpaceCoordinate mc1_input_coord = mk_pt_coord(0_n, 0_n, 0_n, 0_n);
    ParallelTensorSpaceCoordinate mc1_weight_coord = mk_pt_coord(0_n, 1_n, 2_n, 0_n);
    ParallelTensorSpaceCoordinate mc1_output_1_coord = mk_pt_coord(1_n, 0_n, 0_n, 1_n);
    ParallelTensorSpaceCoordinate mc1_output_2_coord = mk_pt_coord(3_n, 0_n, 0_n, 0_n);

    ParallelTensorSpaceCoordinate mc2_input_coord = mk_pt_coord(0_n, 1_n, 0_n, 0_n);
    ParallelTensorSpaceCoordinate mc2_weight_coord = mk_pt_coord(0_n, 4_n, 2_n, 0_n);
    ParallelTensorSpaceCoordinate mc2_output_1_coord = mk_pt_coord(1_n, 2_n, 0_n, 1_n);
    ParallelTensorSpaceCoordinate mc2_output_2_coord = mk_pt_coord(0_n, 0_n, 0_n, 0_n);

    MappedOperatorTaskGroup mapped_task_group = MappedOperatorTaskGroup{
      bidict<MachineSpaceCoordinate, OperatorAtomicTaskShardBinding>{
        {
          mc1,
          mk_shard_binding(
            mc1_input_coord,
            mc1_weight_coord,
            mc1_output_1_coord,
            mc1_output_2_coord),
        },
        {
          mc2,
          mk_shard_binding(
            mc2_input_coord,
            mc2_weight_coord,
            mc2_output_1_coord,
            mc2_output_2_coord),
        },
      },
    };

    auto mk_slot = [](TensorSlotName const &slot_name) 
      -> DynamicTensorSlot
    {
      return DynamicTensorSlot{
        /*slot_name=*/slot_name,
        /*slot_tensor_role=*/std::nullopt,
      };
    };

    auto mk_value = [](size_t src_node_id, 
                       TensorSlotName src_slot_name, 
                       std::optional<ParallelTensorSpaceCoordinate> const &shard_coord) 
      -> DynamicValueAttrs
    {
      return DynamicValueAttrs{
        /*pcg_tensor_guid=*/parallel_tensor_guid_t{
          KwargDataflowOutput<TensorSlotName>{
            Node{src_node_id},
            src_slot_name,
          },
        },
        /*parallel_tensor_shape=*/std::nullopt,
        /*shard_coord=*/shard_coord,
        /*accessor=*/std::nullopt,
        /*role=*/std::nullopt,
      };
    };

    DynamicNodeInvocation input = DynamicNodeInvocation{
      /*inputs=*/{
        {
          mk_slot(TensorSlotName::INPUT),
          mk_value(0, TensorSlotName::OUTPUT, std::nullopt),
        },
        {
          mk_slot(TensorSlotName::WEIGHT),
          mk_value(1, TensorSlotName::OUTPUT, std::nullopt),
        },
      },
      /*node_attrs=*/DynamicNodeAttrs{
        /*task_type=*/std::nullopt,
        /*device_coord=*/std::nullopt,
        /*mapping=*/mapped_task_group,
        /*op_attrs=*/std::nullopt,
        /*pcg_layer_guid=*/parallel_layer_guid_t{Node{20}},
        /*per_device_op_state=*/std::nullopt,
      },
      /*outputs=*/{
        {
          mk_slot(TensorSlotName::OUTPUT_1),
          mk_value(20, TensorSlotName::OUTPUT_1, std::nullopt),
        },
        {
          mk_slot(TensorSlotName::OUTPUT_2),
          mk_value(20, TensorSlotName::OUTPUT_2, std::nullopt),
        },
      },
    };

    std::unordered_set<DynamicNodeInvocation> result = 
      perform_shard_expansion_for_invocation(input);

    auto mk_invocation_shard = [&](
        MachineSpaceCoordinate const &device_coord,
        ParallelTensorSpaceCoordinate const &input_shard_coord,
        ParallelTensorSpaceCoordinate const &weight_shard_coord,
        ParallelTensorSpaceCoordinate const &output_1_shard_coord,
        ParallelTensorSpaceCoordinate const &output_2_shard_coord
      ) ->  DynamicNodeInvocation
    {
      return  DynamicNodeInvocation{
        /*inputs=*/{
          {
            mk_slot(TensorSlotName::INPUT),
            mk_value(0, TensorSlotName::OUTPUT, input_shard_coord),
          },
          {
            mk_slot(TensorSlotName::WEIGHT),
            mk_value(1, TensorSlotName::OUTPUT, weight_shard_coord),
          },
        },
        /*node_attrs=*/DynamicNodeAttrs{
          /*task_type=*/std::nullopt,
          /*device_coord=*/device_coord,
          /*mapping=*/mapped_task_group,
          /*op_attrs=*/std::nullopt,
          /*pcg_layer_guid=*/parallel_layer_guid_t{Node{20}},
          /*per_device_op_state=*/std::nullopt,
        },
        /*outputs=*/{
          {
            mk_slot(TensorSlotName::OUTPUT_1),
            mk_value(20, TensorSlotName::OUTPUT_1, output_1_shard_coord),
          },
          {
            mk_slot(TensorSlotName::OUTPUT_2),
            mk_value(20, TensorSlotName::OUTPUT_2, output_2_shard_coord),
          },
        },
      };
    };

    std::unordered_set<DynamicNodeInvocation> correct = {
      mk_invocation_shard(
        mc1, 
        mc1_input_coord,
        mc1_weight_coord,
        mc1_output_1_coord,
        mc1_output_2_coord),
      mk_invocation_shard(
        mc2, 
        mc2_input_coord,
        mc2_weight_coord,
        mc2_output_1_coord,
        mc2_output_2_coord),
    };

    CHECK(result.size() == correct.size());
    CHECK(result == correct);
  }
}
