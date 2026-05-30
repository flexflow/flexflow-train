#include "task-spec/dynamic_graph/copy_insertion.h"
#include "op-attrs/tensor_slot_name.dtg.h"
#include "pcg/mapped_parallel_computation_graph/mapped_operator_task_group.h"
#include "task-spec/dynamic_graph/dynamic_open_dataflow_graph.h"
#include "task-spec/dynamic_graph/dynamic_task_type.dtg.h"
#include "task-spec/dynamic_graph/dynamic_tensor_role.h"
#include "task-spec/dynamic_graph/dynamic_value_attrs.dtg.h"
#include "test/utils/doctest/check_kv.h"
#include "test/utils/doctest/fmt/unordered_set.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("perform_copy_insertion") {

    auto mk_slot = [](TensorSlotName tensor_slot_name) -> DynamicTensorSlot {
      return DynamicTensorSlot{
          tensor_slot_name,
          std::nullopt,
      };
    };

    auto mk_value_attrs =
        [](size_t src_layer_guid,
           TensorSlotName src_slot,
           std::optional<ParallelTensorMapping> const &mapping)
        -> DynamicValueAttrs {
      return DynamicValueAttrs{
          /*tensor_guid=*/dynamic_tensor_guid_t{
              parallel_tensor_guid_t{
                  KwargDataflowOutput{
                      Node{
                          src_layer_guid,
                      },
                      src_slot,
                  },
              },
          },
          /*parallel_tensor_shape=*/std::nullopt,
          /*shard_coord=*/std::nullopt,
          /*mapping=*/mapping,
          /*accessor=*/std::nullopt,
          /*role=*/std::nullopt,
      };
    };

    auto mk_ptensor_coord =
        [](nonnegative_int shard_idx) -> ParallelTensorSpaceCoordinate {
      return ParallelTensorSpaceCoordinate{
          /*sum_component=*/0_n,
          /*discard_copy_component=*/0_n,
          /*shard_components=*/
          FFOrdered<nonnegative_int>{
              shard_idx,
          },
      };
    };

    auto mk_machine_coord =
        [](nonnegative_int device_idx) -> MachineSpaceCoordinate {
      return MachineSpaceCoordinate{
          /*node_idx=*/0_n,
          /*device_idx=*/device_idx,
      };
    };

    auto mk_device_id = [&](nonnegative_int device_idx) -> device_id_t {
      return device_id_t{
          mk_machine_coord(device_idx),
          DeviceType::GPU,
      };
    };

    auto mk_pcg_layer_guid = [](size_t pcg_layer_guid) -> dynamic_layer_guid_t {
      return dynamic_layer_guid_t{
          parallel_layer_guid_t{
              Node{pcg_layer_guid},
          },
      };
    };

    auto mk_node_attrs =
        [](dynamic_layer_guid_t layer_guid,
           std::optional<DynamicNodeMapping> const &mapping,
           std::optional<TrainingOperationAttrs> const &op_attrs)
        -> DynamicNodeAttrs {
      return DynamicNodeAttrs{
          /*task_type=*/std::nullopt,
          /*device_coord=*/std::nullopt,
          /*mapping=*/mapping,
          /*op_attrs=*/op_attrs,
          /*layer_guid=*/layer_guid,
          /*per_device_op_state=*/std::nullopt,
      };
    };

    auto mk_binding = [&](nonnegative_int input_shard_idx,
                          nonnegative_int output_shard_idx)
        -> OperatorAtomicTaskShardBinding {
      return OperatorAtomicTaskShardBinding{
          /*tensor_coords=*/{
              {
                  TensorSlotName::INPUT,
                  mk_ptensor_coord(input_shard_idx),
              },
              {
                  TensorSlotName::OUTPUT,
                  mk_ptensor_coord(output_shard_idx),
              },
          },
      };
    };

    DynamicValueAttrs v1 = mk_value_attrs(
        /*src_layer_guid=*/0,
        /*src_slot=*/TensorSlotName::OUTPUT,
        /*mapping=*/std::nullopt);

    DynamicValueAttrs v2 = mk_value_attrs(
        /*src_layer_guid=*/1,
        /*src_slot=*/TensorSlotName::OUTPUT,
        /*mapping=*/std::nullopt);

    DynamicValueAttrs v3 = mk_value_attrs(
        /*src_layer_guid=*/2,
        /*src_slot=*/TensorSlotName::OUTPUT,
        /*mapping=*/std::nullopt);

    SUBCASE("inserts copy when necessary") {
      DynamicNodeMapping mapping1 = DynamicNodeMapping{
          MappedOperatorTaskGroup{
              bidict<MachineSpaceCoordinate, OperatorAtomicTaskShardBinding>{
                  {
                      mk_machine_coord(0_n),
                      mk_binding(0_n, 0_n),
                  },
                  {
                      mk_machine_coord(1_n),
                      mk_binding(1_n, 1_n),
                  },
              },
          },
          DeviceType::GPU,
      };

      DynamicNodeMapping mapping2 = DynamicNodeMapping{
          MappedOperatorTaskGroup{
              bidict<MachineSpaceCoordinate, OperatorAtomicTaskShardBinding>{
                  {
                      mk_machine_coord(0_n),
                      mk_binding(0_n, 0_n),
                  },
                  {
                      mk_machine_coord(2_n),
                      mk_binding(1_n, 1_n),
                  },
              },
          },
          DeviceType::GPU,
      };

      DynamicNodeInvocation inv1 = DynamicNodeInvocation{
          /*inputs=*/{
              {
                  mk_slot(TensorSlotName::INPUT),
                  v1,
              },
          },
          /*node_attrs=*/
          mk_node_attrs(
              mk_pcg_layer_guid(1), mapping1, /*op_attrs=*/std::nullopt),
          /*outputs=*/
          {
              {
                  mk_slot(TensorSlotName::OUTPUT),
                  v2,
              },
          },
      };

      DynamicNodeInvocation inv2 = DynamicNodeInvocation{
          /*inputs=*/{
              {
                  mk_slot(TensorSlotName::INPUT),
                  v2,
              },
          },
          /*node_attrs=*/
          mk_node_attrs(
              mk_pcg_layer_guid(2), mapping2, /*op_attrs=*/std::nullopt),
          /*outputs=*/
          {
              {
                  mk_slot(TensorSlotName::OUTPUT),
                  v3,
              },
          },
      };

      DynamicOpenDataflowGraph g =
          dynamic_open_dataflow_graph_from_invocation_set({inv1, inv2});

      DynamicOpenDataflowGraph result = perform_copy_insertion(g);

      DynamicOpenDataflowGraph correct = [&] {
        DynamicValueAttrs mapped_v1 = mk_value_attrs(
            /*src_layer_guid=*/0,
            /*src_slot=*/TensorSlotName::OUTPUT,
            /*mapping=*/
            ParallelTensorMapping{
                bidict<ParallelTensorSpaceCoordinate, device_id_t>{
                    {mk_ptensor_coord(0_n), mk_device_id(0_n)},
                    {mk_ptensor_coord(1_n), mk_device_id(1_n)},
                },
            });

        DynamicValueAttrs mapped_v2_placement1 = mk_value_attrs(
            /*src_layer_guid=*/1,
            /*src_slot=*/TensorSlotName::OUTPUT,
            /*mapping=*/
            ParallelTensorMapping{
                bidict<ParallelTensorSpaceCoordinate, device_id_t>{
                    {mk_ptensor_coord(0_n), mk_device_id(0_n)},
                    {mk_ptensor_coord(1_n), mk_device_id(1_n)},
                },
            });

        DynamicValueAttrs mapped_v2_placement2 = mk_value_attrs(
            /*src_layer_guid=*/1,
            /*src_slot=*/TensorSlotName::OUTPUT,
            /*mapping=*/
            ParallelTensorMapping{
                bidict<ParallelTensorSpaceCoordinate, device_id_t>{
                    {mk_ptensor_coord(0_n), mk_device_id(0_n)},
                    {mk_ptensor_coord(1_n), mk_device_id(2_n)},
                },
            });

        DynamicValueAttrs mapped_v3 = mk_value_attrs(
            /*src_layer_guid=*/2,
            /*src_slot=*/TensorSlotName::OUTPUT,
            /*mapping=*/
            ParallelTensorMapping{
                bidict<ParallelTensorSpaceCoordinate, device_id_t>{
                    {mk_ptensor_coord(0_n), mk_device_id(0_n)},
                    {mk_ptensor_coord(1_n), mk_device_id(2_n)},
                },
            });

        DynamicNodeInvocation mapped_inv1 = DynamicNodeInvocation{
            /*inputs=*/{
                {
                    mk_slot(TensorSlotName::INPUT),
                    mapped_v1,
                },
            },
            /*node_attrs=*/
            mk_node_attrs(
                mk_pcg_layer_guid(1), mapping1, /*op_attrs=*/std::nullopt),
            /*outputs=*/
            {
                {
                    mk_slot(TensorSlotName::OUTPUT),
                    mapped_v2_placement1,
                },
            },
        };

        DynamicNodeInvocation inserted_copy = DynamicNodeInvocation{
            /*inputs=*/{
                {
                    mk_slot(TensorSlotName::INPUT),
                    mapped_v2_placement1,
                },
            },
            /*node_attrs=*/
            mk_node_attrs(dynamic_layer_guid_t{dynamic_copy_layer_guid_t{}},
                          std::nullopt,
                          /*op_attrs=*/TrainingOperationAttrs{CopyAttrs{}}),
            /*outputs=*/
            {
                {
                    mk_slot(TensorSlotName::OUTPUT),
                    mapped_v2_placement2,
                },
            },

        };

        DynamicNodeInvocation mapped_inv2 = DynamicNodeInvocation{
            /*inputs=*/{
                {
                    mk_slot(TensorSlotName::INPUT),
                    mapped_v2_placement2,
                },
            },
            /*node_attrs=*/
            mk_node_attrs(
                mk_pcg_layer_guid(2), mapping2, /*op_attrs=*/std::nullopt),
            /*outputs=*/
            {
                {
                    mk_slot(TensorSlotName::OUTPUT),
                    mapped_v3,
                },
            },
        };

        return dynamic_open_dataflow_graph_from_invocation_set(
            {mapped_inv1, mapped_inv2, inserted_copy});
      }();

      CHECK_MESSAGE(
          result == correct,
          check_kv("result\n", dynamic_open_dataflow_graph_as_dot(result)),
          check_kv("correct\n", dynamic_open_dataflow_graph_as_dot(correct)));
    }

    SUBCASE("does not insert a copy when not necessary") {
      DynamicNodeMapping mapping1 = DynamicNodeMapping{
          MappedOperatorTaskGroup{
              bidict<MachineSpaceCoordinate, OperatorAtomicTaskShardBinding>{
                  {
                      mk_machine_coord(0_n),
                      mk_binding(0_n, 0_n),
                  },
                  {
                      mk_machine_coord(1_n),
                      mk_binding(1_n, 1_n),
                  },
              },
          },
          DeviceType::GPU,
      };

      DynamicNodeMapping mapping2 = DynamicNodeMapping{
          MappedOperatorTaskGroup{
              bidict<MachineSpaceCoordinate, OperatorAtomicTaskShardBinding>{
                  {
                      mk_machine_coord(0_n),
                      mk_binding(0_n, 0_n),
                  },
                  {
                      mk_machine_coord(1_n),
                      mk_binding(1_n, 1_n),
                  },
              },
          },
          DeviceType::GPU,
      };

      DynamicNodeInvocation inv1 = DynamicNodeInvocation{
          /*inputs=*/{
              {
                  mk_slot(TensorSlotName::INPUT),
                  v1,
              },
          },
          /*node_attrs=*/
          mk_node_attrs(
              mk_pcg_layer_guid(1), mapping1, /*op_attrs=*/std::nullopt),
          /*outputs=*/
          {
              {
                  mk_slot(TensorSlotName::OUTPUT),
                  v2,
              },
          },
      };

      DynamicNodeInvocation inv2 = DynamicNodeInvocation{
          /*inputs=*/{
              {
                  mk_slot(TensorSlotName::INPUT),
                  v2,
              },
          },
          /*node_attrs=*/
          mk_node_attrs(
              mk_pcg_layer_guid(2), mapping2, /*op_attrs=*/std::nullopt),
          /*outputs=*/
          {
              {
                  mk_slot(TensorSlotName::OUTPUT),
                  v3,
              },
          },
      };

      DynamicOpenDataflowGraph g =
          dynamic_open_dataflow_graph_from_invocation_set({inv1, inv2});

      DynamicOpenDataflowGraph result = perform_copy_insertion(g);

      DynamicOpenDataflowGraph correct = [&] {
        DynamicValueAttrs mapped_v1 = mk_value_attrs(
            /*src_layer_guid=*/0,
            /*src_slot=*/TensorSlotName::OUTPUT,
            /*mapping=*/
            ParallelTensorMapping{
                bidict<ParallelTensorSpaceCoordinate, device_id_t>{
                    {mk_ptensor_coord(0_n), mk_device_id(0_n)},
                    {mk_ptensor_coord(1_n), mk_device_id(1_n)},
                },
            });

        DynamicValueAttrs mapped_v2 = mk_value_attrs(
            /*src_layer_guid=*/1,
            /*src_slot=*/TensorSlotName::OUTPUT,
            /*mapping=*/
            ParallelTensorMapping{
                bidict<ParallelTensorSpaceCoordinate, device_id_t>{
                    {mk_ptensor_coord(0_n), mk_device_id(0_n)},
                    {mk_ptensor_coord(1_n), mk_device_id(1_n)},
                },
            });

        DynamicValueAttrs mapped_v3 = mk_value_attrs(
            /*src_layer_guid=*/2,
            /*src_slot=*/TensorSlotName::OUTPUT,
            /*mapping=*/
            ParallelTensorMapping{
                bidict<ParallelTensorSpaceCoordinate, device_id_t>{
                    {mk_ptensor_coord(0_n), mk_device_id(0_n)},
                    {mk_ptensor_coord(1_n), mk_device_id(1_n)},
                },
            });

        DynamicNodeInvocation mapped_inv1 = DynamicNodeInvocation{
            /*inputs=*/{
                {
                    mk_slot(TensorSlotName::INPUT),
                    mapped_v1,
                },
            },
            /*node_attrs=*/
            mk_node_attrs(
                mk_pcg_layer_guid(1), mapping1, /*op_attrs=*/std::nullopt),
            /*outputs=*/
            {
                {
                    mk_slot(TensorSlotName::OUTPUT),
                    mapped_v2,
                },
            },
        };

        DynamicNodeInvocation mapped_inv2 = DynamicNodeInvocation{
            /*inputs=*/{
                {
                    mk_slot(TensorSlotName::INPUT),
                    mapped_v2,
                },
            },
            /*node_attrs=*/
            mk_node_attrs(
                mk_pcg_layer_guid(2), mapping2, /*op_attrs=*/std::nullopt),
            /*outputs=*/
            {
                {
                    mk_slot(TensorSlotName::OUTPUT),
                    mapped_v3,
                },
            },
        };

        return dynamic_open_dataflow_graph_from_invocation_set(
            {mapped_inv1, mapped_inv2});
      }();

      CHECK_MESSAGE(
          result == correct,
          check_kv("result\n", dynamic_open_dataflow_graph_as_dot(result)),
          check_kv("correct\n", dynamic_open_dataflow_graph_as_dot(correct)));
    }
  }
}
