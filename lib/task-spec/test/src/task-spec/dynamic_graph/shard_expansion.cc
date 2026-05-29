#include "task-spec/dynamic_graph/shard_expansion.h"
#include "pcg/mapped_parallel_computation_graph/mapped_operator_task_group.h"
#include "task-spec/dynamic_graph/copy_attrs.dtg.h"
#include "task-spec/dynamic_graph/dynamic_copy_layer_guid_t.dtg.h"
#include "task-spec/dynamic_graph/training_operation_attrs.dtg.h"
#include "test/utils/doctest/fmt/unordered_set.h"
#include <doctest/doctest.h>
#include "task-spec/dynamic_graph/dynamic_tensor_role.h"
#include "op-attrs/ops/element_unary.h"
#include "utils/one_to_many/one_to_many_filter_keys.h"
#include "utils/one_to_many/one_to_many_filter_values.h"
#include "utils/containers/map_from_pairs.h"
#include "utils/containers/binary_merge_disjoint_maps.h"

using namespace ::FlexFlow;

static MachineSpaceCoordinate mk_machine_coord(nonnegative_int node_idx,
                                               nonnegative_int device_idx) {
  return MachineSpaceCoordinate{
      /*node_idx=*/node_idx,
      /*device_idx=*/device_idx,
      /*device_type=*/DeviceType::GPU,
  };
};

static ParallelTensorSpaceCoordinate mk_pt_coord(nonnegative_int idx1,
                                                 nonnegative_int idx2,
                                                 nonnegative_int idx3,
                                                 nonnegative_int idx4) {
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

DynamicTensorSlot mk_slot(TensorSlotName const &slot_name,
                          std::optional<MachineSpaceCoordinate> const &task_shard = std::nullopt) {
  return DynamicTensorSlot{
      /*slot_name=*/slot_name,
      /*slot_tensor_role=*/std::nullopt,
      /*task_shard=*/task_shard,
  };
};

DynamicValueAttrs
    mk_value(size_t src_node_id,
             TensorSlotName src_slot_name,
             OneToMany<ParallelTensorSpaceCoordinate, MachineSpaceCoordinate> const &tensor_binding,
             std::optional<ParallelTensorSpaceCoordinate> const &shard_coord,
             std::optional<DynamicTensorRole> const &role = std::nullopt) {

  OneToMany<ParallelTensorSpaceCoordinate, MachineSpaceCoordinate> mapping = tensor_binding;
  if (shard_coord.has_value()) {
    mapping = one_to_many_filter_keys(mapping,
                                      [&](ParallelTensorSpaceCoordinate const &p) {
                                        return p == shard_coord.value();
                                      });
  }

  return DynamicValueAttrs{
      /*tensor_guid=*/dynamic_tensor_guid_t{parallel_tensor_guid_t{
          KwargDataflowOutput<TensorSlotName>{
              Node{src_node_id},
              src_slot_name,
          },
      }},
      /*parallel_tensor_shape=*/std::nullopt,
      /*shard_coord=*/shard_coord,
      /*mapping=*/mapping,
      /*accessor=*/std::nullopt,
      /*role=*/role,
  };
};

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("generate_shard_expansion_for_invocation") {
    auto mk_op_value =
        [&](size_t src_node_id,
            TensorSlotName src_slot_name,
            TensorSlotName use_slot_name,
            MappedOperatorTaskGroup const &mapped_task_group,
            std::optional<ParallelTensorSpaceCoordinate> const &shard_coord,
            std::optional<DynamicTensorRole> const &role = std::nullopt)
        -> DynamicValueAttrs {
      OneToMany<ParallelTensorSpaceCoordinate, MachineSpaceCoordinate>
          tensor_binding = get_tensor_bindings_for_slot_name(mapped_task_group,
                                                             use_slot_name);
      return mk_value(src_node_id, src_slot_name, tensor_binding, shard_coord, role);
    };

    auto mk_sharding_info = [&](TensorSlotName slot_name,
                                ParallelTensorSpaceCoordinate const &shard_coord,
                                MappedOperatorTaskGroup const &mapped_op_task_group,
                                MachineSpaceCoordinate const &device_coord)
      -> std::pair<DynamicTensorSlot, DynamicValueAttrsShardingInfo>
    {
      OneToMany<ParallelTensorSpaceCoordinate, MachineSpaceCoordinate>
          tensor_binding = get_tensor_bindings_for_slot_name(mapped_op_task_group,
                                                             slot_name);
      return std::pair{
        mk_slot(slot_name),
        DynamicValueAttrsShardingInfo{
          /*shard_coord=*/shard_coord,
          /*mapping=*/one_to_many_filter_values(tensor_binding,
                                                [&](MachineSpaceCoordinate const &c) -> bool {
                                                  return device_coord == c;
                                                }),
        },
      };
    };

    SUBCASE("standard operator") {
      MachineSpaceCoordinate mc1 = mk_machine_coord(0_n, 0_n);
      MachineSpaceCoordinate mc2 = mk_machine_coord(2_n, 0_n);

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

      TrainingOperationAttrs op_attrs = TrainingOperationAttrs{
        PCGOperatorAttrs{
          make_relu_attrs(),
        },
      };

      MappedOperatorTaskGroup mapped_task_group = MappedOperatorTaskGroup{
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

      DynamicNodeInvocation input = DynamicNodeInvocation{
          /*inputs=*/{
              {
                  mk_slot(TensorSlotName::INPUT),
                  mk_op_value(0,
                              TensorSlotName::OUTPUT,
                              TensorSlotName::INPUT,
                              mapped_task_group,
                              std::nullopt),
              },
              {
                  mk_slot(TensorSlotName::WEIGHT),
                  mk_op_value(1,
                              TensorSlotName::OUTPUT,
                              TensorSlotName::WEIGHT,
                              mapped_task_group,
                              std::nullopt),
              },
          },
          /*node_attrs=*/
          DynamicNodeAttrs{
              /*task_type=*/std::nullopt,
              /*device_coord=*/std::nullopt,
              /*mapping=*/mapped_task_group,
              /*op_attrs=*/op_attrs,
              /*layer_guid=*/
              dynamic_layer_guid_t{parallel_layer_guid_t{Node{20}}},
              /*per_device_op_state=*/std::nullopt,
          },
          /*outputs=*/
          {
              {
                  mk_slot(TensorSlotName::OUTPUT_1),
                  mk_op_value(20,
                              TensorSlotName::OUTPUT_1,
                              TensorSlotName::OUTPUT_1,
                              mapped_task_group,
                              std::nullopt),
              },
              {
                  mk_slot(TensorSlotName::OUTPUT_2),
                  mk_op_value(20,
                              TensorSlotName::OUTPUT_2,
                              TensorSlotName::OUTPUT_2,
                              mapped_task_group,
                              std::nullopt),
              },
          },
      };

      std::unordered_set<DynamicNodeInvocationShardingInfo> result =
          generate_shard_expansion_for_invocation(input);

      auto mk_invocation_shard =
          [&](MachineSpaceCoordinate const &device_coord,
              ParallelTensorSpaceCoordinate const &input_shard_coord,
              ParallelTensorSpaceCoordinate const &weight_shard_coord,
              ParallelTensorSpaceCoordinate const &output_1_shard_coord,
              ParallelTensorSpaceCoordinate const &output_2_shard_coord)
          -> DynamicNodeInvocationShardingInfo {
        return DynamicNodeInvocationShardingInfo{
          /*device_coord=*/nonempty_set{device_coord},
          /*value_sharding=*/{
            mk_sharding_info(TensorSlotName::INPUT, input_shard_coord, mapped_task_group, device_coord),
            mk_sharding_info(TensorSlotName::WEIGHT, weight_shard_coord, mapped_task_group, device_coord),
            mk_sharding_info(TensorSlotName::OUTPUT_1, output_1_shard_coord, mapped_task_group, device_coord),
            mk_sharding_info(TensorSlotName::OUTPUT_2, output_2_shard_coord, mapped_task_group, device_coord),
          },
        };
      };

      std::unordered_set<DynamicNodeInvocationShardingInfo> correct = {
          mk_invocation_shard(mc1,
                              mc1_input_coord,
                              mc1_weight_coord,
                              mc1_output_1_coord,
                              mc1_output_2_coord),
          mk_invocation_shard(mc2,
                              mc2_input_coord,
                              mc2_weight_coord,
                              mc2_output_1_coord,
                              mc2_output_2_coord),
      };

      nlohmann::json result_json = result;
      nlohmann::json correct_json = correct;

      CHECK(result.size() == correct.size());
      CHECK(result_json == correct_json);
      CHECK(result == correct);
    }

    SUBCASE("copy operator") {
      MachineSpaceCoordinate mc1 = mk_machine_coord(0_n, 0_n);
      MachineSpaceCoordinate mc2 = mk_machine_coord(1_n, 0_n);
      MachineSpaceCoordinate mc3 = mk_machine_coord(2_n, 0_n);
      MachineSpaceCoordinate mc4 = mk_machine_coord(3_n, 0_n);

      ParallelTensorSpaceCoordinate pt1 = mk_pt_coord(0_n, 0_n, 0_n, 0_n);
      ParallelTensorSpaceCoordinate pt2 = mk_pt_coord(0_n, 1_n, 0_n, 0_n);

      OneToMany<ParallelTensorSpaceCoordinate, MachineSpaceCoordinate> src_binding{
          {pt1, {mc1}},
          {pt2, {mc2}},
      };

      OneToMany<ParallelTensorSpaceCoordinate, MachineSpaceCoordinate> dst_binding{
          {pt1, {mc3}},
          {pt2, {mc4}},
      };

      DynamicNodeInvocation input = DynamicNodeInvocation{
          /*inputs=*/{
              {
                  mk_slot(TensorSlotName::INPUT),
                  mk_value(0, TensorSlotName::OUTPUT, src_binding, std::nullopt),
              },
          },
          /*node_attrs=*/
          DynamicNodeAttrs{
              /*task_type=*/std::nullopt,
              /*device_coord=*/std::nullopt,
              /*mapping=*/std::nullopt,
              /*op_attrs=*/TrainingOperationAttrs{CopyAttrs{}},
              /*layer_guid=*/dynamic_layer_guid_t{dynamic_copy_layer_guid_t{}},
              /*per_device_op_state=*/std::nullopt,
          },
          /*outputs=*/
          {
              {
                  mk_slot(TensorSlotName::OUTPUT),
                  mk_value(20, TensorSlotName::OUTPUT, dst_binding, std::nullopt),
              },
          },
      };

      std::unordered_set<DynamicNodeInvocationShardingInfo> result =
          generate_shard_expansion_for_invocation(input);

      auto mk_invocation_shard =
          [&](MachineSpaceCoordinate const &device_coord,
              ParallelTensorSpaceCoordinate const &tensor_shard_coord)
          -> DynamicNodeInvocationShardingInfo {

        return DynamicNodeInvocationShardingInfo{
          /*device_coord=*/nonempty_set{device_coord},
          /*value_sharding=*/std::map<DynamicTensorSlot, DynamicValueAttrsShardingInfo>{
            {
              mk_slot(TensorSlotName::INPUT),
              DynamicValueAttrsShardingInfo{
                tensor_shard_coord,
                one_to_many_filter_keys(src_binding,
                                        [&](ParallelTensorSpaceCoordinate const &pt_coord) -> bool {
                                          return pt_coord == tensor_shard_coord;
                                        }),
              },
            },
            {
              mk_slot(TensorSlotName::OUTPUT),
              DynamicValueAttrsShardingInfo{
                tensor_shard_coord,
                one_to_many_filter_keys(dst_binding,
                                        [&](ParallelTensorSpaceCoordinate const &pt_coord) -> bool {
                                          return pt_coord == tensor_shard_coord;
                                        }),
              },
            },
          },
        };
      };

      std::unordered_set<DynamicNodeInvocationShardingInfo> correct = {
          mk_invocation_shard(mc1, pt1),
          mk_invocation_shard(mc2, pt2),
      };

      CHECK(result.size() == correct.size());
      CHECK(result == correct);
    }

    SUBCASE("replicate operator") {
      MachineSpaceCoordinate mc1 = mk_machine_coord(0_n, 0_n);
      MachineSpaceCoordinate mc2 = mk_machine_coord(1_n, 0_n);
      MachineSpaceCoordinate mc3 = mk_machine_coord(2_n, 0_n);
      MachineSpaceCoordinate mc4 = mk_machine_coord(3_n, 0_n);

      ParallelTensorSpaceCoordinate pt1 = mk_pt_coord(0_n, 0_n, 0_n, 0_n);
      ParallelTensorSpaceCoordinate pt2 = mk_pt_coord(0_n, 0_n, 0_n, 1_n);
      ParallelTensorSpaceCoordinate pt3 = mk_pt_coord(0_n, 1_n, 0_n, 0_n);
      ParallelTensorSpaceCoordinate pt4 = mk_pt_coord(0_n, 1_n, 0_n, 1_n);

      auto mk_shard_binding = [&](ParallelTensorSpaceCoordinate const &c1,
                                  ParallelTensorSpaceCoordinate const &c2)
          -> OperatorAtomicTaskShardBinding {
        return OperatorAtomicTaskShardBinding{
            /*tensor_coords=*/{
                {
                    TensorSlotName::INPUT,
                    c1,
                },
                {
                    TensorSlotName::OUTPUT,
                    c2,
                },
            },
        };
      };

      MappedOperatorTaskGroup mapped_task_group = MappedOperatorTaskGroup{
          bidict<MachineSpaceCoordinate, OperatorAtomicTaskShardBinding>{
              {
                  mc1,
                  mk_shard_binding(pt1, pt1),
              },
              {
                  mc2,
                  mk_shard_binding(pt1, pt2),
              },
              {
                  mc3,
                  mk_shard_binding(pt2, pt3),
              },
              {
                  mc4,
                  mk_shard_binding(pt2, pt4),
              },
          },
      };

      SUBCASE("fwd") {
        OneToMany<ParallelTensorSpaceCoordinate, MachineSpaceCoordinate> src_binding{
            {pt1, {mc1}},
            {pt2, {mc2}},
        };

        OneToMany<ParallelTensorSpaceCoordinate, MachineSpaceCoordinate> dst_binding{
            {pt1, {mc1}},
            {pt2, {mc2}},
            {pt3, {mc3}},
            {pt4, {mc4}},
        };

        DynamicNodeInvocation input = DynamicNodeInvocation{
            /*inputs=*/{
                {
                    DynamicTensorSlot{
                      /*slot_name=*/TensorSlotName::INPUT,
                      /*slot_tensor_role=*/mk_dynamic_tensor_role_fwd(),
                      /*task_shard=*/std::nullopt,
                    },
                    mk_value(0, TensorSlotName::OUTPUT, src_binding, std::nullopt),
                },
            },
            /*node_attrs=*/
            DynamicNodeAttrs{
              /*task_type=*/DynamicTaskType::FWD,
                /*device_coords=*/std::nullopt,
                /*mapping=*/mapped_task_group,
                /*op_attrs=*/TrainingOperationAttrs{
                  PCGOperatorAttrs{
                    ReplicateAttrs{
                      /*replicate_degree=*/2_p,
                    },
                  },
                },
                /*layer_guid=*/dynamic_layer_guid_t{parallel_layer_guid_t{Node{20}}},
                /*per_device_op_state=*/std::nullopt,
            },
            /*outputs=*/
            {
                {
                    DynamicTensorSlot{
                      /*slot_name=*/TensorSlotName::OUTPUT,
                      /*slot_tensor_role=*/mk_dynamic_tensor_role_fwd(),
                      /*task_shard=*/std::nullopt,
                    },
                    mk_value(20, TensorSlotName::OUTPUT, dst_binding, std::nullopt),
                },
            },
        };

        std::unordered_set<DynamicNodeInvocationShardingInfo> result =
            generate_shard_expansion_for_invocation(input);


        auto mk_output_binding = [&](MachineSpaceCoordinate const &mc)
          -> std::pair<DynamicTensorSlot, DynamicValueAttrsShardingInfo>
        {
          return {
            DynamicTensorSlot{
              /*slot_name=*/TensorSlotName::OUTPUT,
              /*slot_tensor_role=*/mk_dynamic_tensor_role_fwd(),
              /*task_shard=*/mc,
            },
            DynamicValueAttrsShardingInfo{
              dst_binding.at_r(mc),
              one_to_many_filter_keys(dst_binding,
                                      [&](ParallelTensorSpaceCoordinate const &pt_coord) -> bool {
                                        return pt_coord == dst_binding.at_r(mc);
                                      }),
            },
          };
        };

        auto mk_invocation_shard =
            [&](nonempty_set<MachineSpaceCoordinate> const &device_coords,
                ParallelTensorSpaceCoordinate const &input_shard_coord,
                std::unordered_set<MachineSpaceCoordinate> const &output_task_shards)
            -> DynamicNodeInvocationShardingInfo {

          return DynamicNodeInvocationShardingInfo{
            /*device_coords=*/device_coords,
            /*value_sharding=*/
              binary_merge_disjoint_maps(
                std::map<DynamicTensorSlot, DynamicValueAttrsShardingInfo>{
                  {
                    DynamicTensorSlot{
                      /*slot_name=*/TensorSlotName::INPUT,
                      /*slot_tensor_role=*/mk_dynamic_tensor_role_fwd(),
                      /*task_shard=*/std::nullopt,
                    },
                    DynamicValueAttrsShardingInfo{
                      input_shard_coord,
                      one_to_many_filter_keys(
                        src_binding,
                        [&](ParallelTensorSpaceCoordinate const &pt_coord) -> bool {
                          return pt_coord == input_shard_coord;
                        }),
                    },
                  },
                },
                map_from_pairs(transform(output_task_shards, mk_output_binding))),
          };
        };

        std::unordered_set<DynamicNodeInvocationShardingInfo> correct = {
          mk_invocation_shard(nonempty_set{mc1, mc2}, pt1, {mc1, mc2}),
          mk_invocation_shard(nonempty_set{mc3, mc4}, pt2, {mc3, mc4}),
        };

        CHECK(result.size() == correct.size());
        CHECK(result == correct);
      }

      SUBCASE("bwd") {
        OneToMany<ParallelTensorSpaceCoordinate, MachineSpaceCoordinate> output_grad_binding{
            {pt1, {mc1}},
            {pt2, {mc2}},
            {pt3, {mc3}},
            {pt4, {mc4}},
        };

        OneToMany<ParallelTensorSpaceCoordinate, MachineSpaceCoordinate> input_grad_binding{
            {pt1, {mc1}},
            {pt2, {mc2}},
        };

        DynamicNodeInvocation input = DynamicNodeInvocation{
            /*inputs=*/{
                {
                    DynamicTensorSlot{
                      /*slot_name=*/TensorSlotName::OUTPUT,
                      /*slot_tensor_role=*/mk_dynamic_tensor_role_bwd(),
                      /*task_shard=*/std::nullopt,
                    },
                    mk_value(0, TensorSlotName::OUTPUT, output_grad_binding, std::nullopt),
                },
            },
            /*node_attrs=*/
            DynamicNodeAttrs{
              /*task_type=*/DynamicTaskType::BWD,
                /*device_coords=*/std::nullopt,
                /*mapping=*/mapped_task_group,
                /*op_attrs=*/TrainingOperationAttrs{
                  PCGOperatorAttrs{
                    ReplicateAttrs{
                      /*replicate_degree=*/2_p,
                    },
                  },
                },
                /*layer_guid=*/dynamic_layer_guid_t{parallel_layer_guid_t{Node{20}}},
                /*per_device_op_state=*/std::nullopt,
            },
            /*outputs=*/
            {
                {
                    DynamicTensorSlot{
                      /*slot_name=*/TensorSlotName::INPUT,
                      /*slot_tensor_role=*/mk_dynamic_tensor_role_bwd(),
                      /*task_shard=*/std::nullopt,
                    },
                    mk_value(20, TensorSlotName::INPUT, input_grad_binding, std::nullopt),
                },
            },
        };

        std::unordered_set<DynamicNodeInvocationShardingInfo> result =
            generate_shard_expansion_for_invocation(input);

        auto mk_output_grad_binding = [&](MachineSpaceCoordinate const &mc)
          -> std::pair<DynamicTensorSlot, DynamicValueAttrsShardingInfo>
        {
          return {
            DynamicTensorSlot{
              /*slot_name=*/TensorSlotName::OUTPUT,
              /*slot_tensor_role=*/mk_dynamic_tensor_role_bwd(),
              /*task_shard=*/mc,
            },
            DynamicValueAttrsShardingInfo{
              output_grad_binding.at_r(mc),
              one_to_many_filter_keys(output_grad_binding,
                                      [&](ParallelTensorSpaceCoordinate const &pt_coord) -> bool {
                                        return pt_coord == output_grad_binding.at_r(mc);
                                      }),
            },
          };
        };

        auto mk_invocation_shard =
            [&](nonempty_set<MachineSpaceCoordinate> const &device_coords,
                std::unordered_set<MachineSpaceCoordinate> const &output_grad_task_shards,
                ParallelTensorSpaceCoordinate const &input_grad_shard_coord)
            -> DynamicNodeInvocationShardingInfo {

          return DynamicNodeInvocationShardingInfo{
            /*device_coords=*/device_coords,
            /*value_sharding=*/
              binary_merge_disjoint_maps(
                std::map<DynamicTensorSlot, DynamicValueAttrsShardingInfo>{
                  {
                    DynamicTensorSlot{
                      /*slot_name=*/TensorSlotName::INPUT,
                      /*slot_tensor_role=*/mk_dynamic_tensor_role_bwd(),
                      /*task_shard=*/std::nullopt,
                    },
                    DynamicValueAttrsShardingInfo{
                      input_grad_shard_coord,
                      one_to_many_filter_keys(
                        input_grad_binding,
                        [&](ParallelTensorSpaceCoordinate const &pt_coord) -> bool {
                          return pt_coord == input_grad_shard_coord;
                        }),
                    },
                  },
                },
                map_from_pairs(transform(output_grad_task_shards, mk_output_grad_binding))),
          };
        };

        std::unordered_set<DynamicNodeInvocationShardingInfo> correct = {
          mk_invocation_shard(nonempty_set{mc1, mc2}, {mc1, mc2}, pt1),
          mk_invocation_shard(nonempty_set{mc3, mc4}, {mc3, mc4}, pt2),
        };

        CHECK(result.size() == correct.size());
        CHECK(result == correct);
      }
    }
  }
}
