#include "task-spec/dynamic_graph/pass_expansion.h"
#include "task-spec/dynamic_graph/dynamic_open_dataflow_graph.h"
#include "task-spec/dynamic_graph/dynamic_tensor_role.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("perform_fwd_pass_expansion_for_invocation") {
    auto mk_value_attrs =
        [](size_t node_id, std::optional<DynamicTensorRole> const &tensor_role)
        -> DynamicValueAttrs {
      return DynamicValueAttrs{
          /*pcg_tensor_guid=*/parallel_tensor_guid_t{
              KwargDataflowOutput<TensorSlotName>{
                  Node{node_id},
                  TensorSlotName::OUTPUT,
              },
          },
          /*parallel_tensor_shape=*/std::nullopt,
          /*shard_coord=*/std::nullopt,
          /*accessor=*/std::nullopt,
          /*role=*/tensor_role,
      };
    };

    auto mk_slot =
        [](TensorSlotName const &slot_name,
           std::optional<DynamicTensorRole> role) -> DynamicTensorSlot {
      return DynamicTensorSlot{
          /*slot_name=*/slot_name,
          /*slot_tensor_role=*/role,
      };
    };

    parallel_layer_guid_t pcg_layer_guid = parallel_layer_guid_t{Node{20}};

    DynamicNodeInvocation invocation = [&]() -> DynamicNodeInvocation {
      DynamicValueAttrs v1 = mk_value_attrs(0, std::nullopt);
      DynamicValueAttrs v2 = mk_value_attrs(1, std::nullopt);
      DynamicValueAttrs v3 = mk_value_attrs(2, std::nullopt);

      return DynamicNodeInvocation{
          /*inputs=*/{
              {mk_slot(TensorSlotName::INPUT, std::nullopt), v1},
              {mk_slot(TensorSlotName::WEIGHT, std::nullopt), v2},
              {mk_slot(TensorSlotName::BIAS, std::nullopt), v1},
              {mk_slot(TensorSlotName::SCALE, std::nullopt), v1},
          },
          /*node_attrs=*/
          DynamicNodeAttrs{
              /*task_type=*/std::nullopt,
              /*device_coord=*/std::nullopt,
              /*mapping=*/std::nullopt,
              /*op_attrs=*/std::nullopt,
              /*pcg_layer_guid=*/pcg_layer_guid,
              /*per_device_op_state=*/std::nullopt,
          },
          /*outputs=*/
          {
              {mk_slot(TensorSlotName::OUTPUT, std::nullopt), v3},
          },
      };
    }();

    DynamicNodeInvocation result =
        perform_fwd_pass_expansion_for_invocation(invocation);

    DynamicNodeInvocation correct = [&]() -> DynamicNodeInvocation {
      DynamicTensorRole fwd_role = DynamicTensorRole{FwbTensorType::FORWARD};

      DynamicValueAttrs v1_fwd = mk_value_attrs(0, fwd_role);
      DynamicValueAttrs v2_fwd = mk_value_attrs(1, fwd_role);
      DynamicValueAttrs v3_fwd = mk_value_attrs(2, fwd_role);

      return DynamicNodeInvocation{
          /*inputs=*/{
              {mk_slot(TensorSlotName::INPUT, fwd_role), v1_fwd},
              {mk_slot(TensorSlotName::WEIGHT, fwd_role), v2_fwd},
              {mk_slot(TensorSlotName::BIAS, fwd_role), v1_fwd},
              {mk_slot(TensorSlotName::SCALE, fwd_role), v1_fwd},
          },
          /*node_attrs=*/
          DynamicNodeAttrs{
              /*task_type=*/DynamicTaskType::FWD,
              /*device_coord=*/std::nullopt,
              /*mapping=*/std::nullopt,
              /*op_attrs=*/std::nullopt,
              /*pcg_layer_guid=*/pcg_layer_guid,
              /*per_device_op_state=*/std::nullopt,
          },
          /*outputs=*/
          {
              {mk_slot(TensorSlotName::OUTPUT, fwd_role), v3_fwd},
          },
      };
    }();

    ASSERT(result == correct);
  }

  TEST_CASE("perform_bwd_pass_expansion_for_invocation") {
    auto mk_value_attrs =
        [](size_t node_id, std::optional<DynamicTensorRole> const &tensor_role)
        -> DynamicValueAttrs {
      return DynamicValueAttrs{
          /*pcg_tensor_guid=*/parallel_tensor_guid_t{
              KwargDataflowOutput<TensorSlotName>{
                  Node{node_id},
                  TensorSlotName::OUTPUT,
              },
          },
          /*parallel_tensor_shape=*/std::nullopt,
          /*shard_coord=*/std::nullopt,
          /*accessor=*/std::nullopt,
          /*role=*/tensor_role,
      };
    };

    auto mk_slot =
        [](TensorSlotName const &slot_name,
           std::optional<DynamicTensorRole> role) -> DynamicTensorSlot {
      return DynamicTensorSlot{
          /*slot_name=*/slot_name,
          /*slot_tensor_role=*/role,
      };
    };

    parallel_layer_guid_t pcg_layer_guid = parallel_layer_guid_t{Node{20}};

    DynamicNodeInvocation invocation = [&]() -> DynamicNodeInvocation {
      DynamicValueAttrs v1 = mk_value_attrs(0, std::nullopt);
      DynamicValueAttrs v2 = mk_value_attrs(1, std::nullopt);
      DynamicValueAttrs v3 = mk_value_attrs(2, std::nullopt);

      return DynamicNodeInvocation{
          /*inputs=*/{
              {mk_slot(TensorSlotName::INPUT, std::nullopt), v1},
              {mk_slot(TensorSlotName::WEIGHT, std::nullopt), v2},
              {mk_slot(TensorSlotName::BIAS, std::nullopt), v1},
              {mk_slot(TensorSlotName::SCALE, std::nullopt), v1},
          },
          /*node_attrs=*/
          DynamicNodeAttrs{
              /*task_type=*/std::nullopt,
              /*device_coord=*/std::nullopt,
              /*mapping=*/std::nullopt,
              /*op_attrs=*/std::nullopt,
              /*pcg_layer_guid=*/pcg_layer_guid,
              /*per_device_op_state=*/std::nullopt,
          },
          /*outputs=*/
          {
              {mk_slot(TensorSlotName::OUTPUT, std::nullopt), v3},
          },
      };
    }();

    DynamicNodeInvocation result =
        perform_bwd_pass_expansion_for_invocation(invocation);

    DynamicNodeInvocation correct = [&]() -> DynamicNodeInvocation {
      DynamicTensorRole fwd_role = DynamicTensorRole{FwbTensorType::FORWARD};
      DynamicTensorRole grad_role = DynamicTensorRole{FwbTensorType::GRADIENT};

      DynamicValueAttrs v1_fwd = mk_value_attrs(0, fwd_role);
      DynamicValueAttrs v2_fwd = mk_value_attrs(1, fwd_role);
      DynamicValueAttrs v3_fwd = mk_value_attrs(2, fwd_role);
      DynamicValueAttrs v1_grad = mk_value_attrs(0, grad_role);
      DynamicValueAttrs v2_grad = mk_value_attrs(1, grad_role);
      DynamicValueAttrs v3_grad = mk_value_attrs(2, grad_role);

      return DynamicNodeInvocation{
          /*inputs=*/{
              {mk_slot(TensorSlotName::INPUT, fwd_role), v1_fwd},
              {mk_slot(TensorSlotName::WEIGHT, fwd_role), v2_fwd},
              {mk_slot(TensorSlotName::BIAS, fwd_role), v1_fwd},
              {mk_slot(TensorSlotName::SCALE, fwd_role), v1_fwd},
              {mk_slot(TensorSlotName::OUTPUT, fwd_role), v3_fwd},
              {mk_slot(TensorSlotName::OUTPUT, grad_role), v3_grad},
          },
          /*node_attrs=*/
          DynamicNodeAttrs{
              /*pass_type=*/DynamicTaskType::BWD,
              /*device_coord=*/std::nullopt,
              /*mapping=*/std::nullopt,
              /*op_attrs=*/std::nullopt,
              /*pcg_layer_guid=*/pcg_layer_guid,
              /*per_device_op_state=*/std::nullopt,
          },
          /*outputs=*/
          {
              {mk_slot(TensorSlotName::INPUT, grad_role), v1_grad},
              {mk_slot(TensorSlotName::WEIGHT, grad_role), v2_grad},
              {mk_slot(TensorSlotName::BIAS, grad_role), v1_grad},
              {mk_slot(TensorSlotName::SCALE, grad_role), v1_grad},
          },
      };
    }();

    ASSERT(result == correct);
  }

  TEST_CASE("perform_pass_expansion(DynamicOpenDataflowGraph)") {
    auto mk_node_attrs = [](size_t layer_id,
                            std::optional<DynamicTaskType> const &pass_type)
        -> DynamicNodeAttrs {
      return DynamicNodeAttrs{
          /*pass_type=*/pass_type,
          /*device_coord=*/std::nullopt,
          /*mapping=*/std::nullopt,
          /*op_attrs=*/std::nullopt,
          /*pcg_layer_guid=*/parallel_layer_guid_t{Node{layer_id}},
          /*per_device_op_state=*/std::nullopt,
      };
    };

    auto mk_value_attrs =
        [](size_t node_id, std::optional<DynamicTensorRole> const &tensor_type)
        -> DynamicValueAttrs {
      return DynamicValueAttrs{
          /*pcg_tensor_guid=*/parallel_tensor_guid_t{
              KwargDataflowOutput{
                  Node{node_id},
                  TensorSlotName::OUTPUT,
              },
          },
          /*parallel_tensor_shape=*/std::nullopt,
          /*shard_coord=*/std::nullopt,
          /*accessor=*/std::nullopt,
          /*role=*/tensor_type,
      };
    };

    DynamicOpenDataflowGraph input = [&]() -> DynamicOpenDataflowGraph {
      DynamicNodeAttrs n1 = mk_node_attrs(10, std::nullopt);
      DynamicNodeAttrs n2 = mk_node_attrs(11, std::nullopt);

      DynamicValueAttrs v1 = mk_value_attrs(0, std::nullopt);
      DynamicValueAttrs v2 = mk_value_attrs(1, std::nullopt);

      std::unordered_set<DynamicNodeInvocation> invocation_set = {
          DynamicNodeInvocation{
              /*inputs=*/std::unordered_map<DynamicTensorSlot,
                                            DynamicValueAttrs>{},
              /*node_attrs=*/n1,
              /*outputs=*/
              std::unordered_map<DynamicTensorSlot, DynamicValueAttrs>{
                  {
                      DynamicTensorSlot{
                          /*slot_name=*/TensorSlotName::OUTPUT,
                          /*slot_tensor_role=*/std::nullopt,
                      },
                      v1,
                  },
              },
          },
          DynamicNodeInvocation{
              /*inputs=*/std::unordered_map<DynamicTensorSlot,
                                            DynamicValueAttrs>{
                  {DynamicTensorSlot{
                       /*slot_name=*/TensorSlotName::INPUT,
                       /*slot_tensor_role=*/std::nullopt,
                   },
                   v1},
              },
              /*node_attrs=*/n2,
              /*outputs=*/
              std::unordered_map<DynamicTensorSlot, DynamicValueAttrs>{
                  {DynamicTensorSlot{
                       /*slot_name=*/TensorSlotName::OUTPUT,
                       /*slot_tensor_role=*/std::nullopt,
                   },
                   v2},
              },
          },
      };

      return dynamic_open_dataflow_graph_from_invocation_set(invocation_set);
    }();

    DynamicOpenDataflowGraph result = perform_pass_expansion(input);

    DynamicOpenDataflowGraph correct = [&]() -> DynamicOpenDataflowGraph {
      DynamicNodeAttrs n1_fwd = mk_node_attrs(10, DynamicTaskType::FWD);
      DynamicNodeAttrs n2_fwd = mk_node_attrs(11, DynamicTaskType::FWD);
      DynamicNodeAttrs n1_bwd = mk_node_attrs(10, DynamicTaskType::BWD);
      DynamicNodeAttrs n2_bwd = mk_node_attrs(11, DynamicTaskType::BWD);

      DynamicValueAttrs v1_activation =
          mk_value_attrs(0, mk_dynamic_tensor_role_fwd());
      DynamicValueAttrs v1_gradient =
          mk_value_attrs(0, mk_dynamic_tensor_role_bwd());
      DynamicValueAttrs v2_activation =
          mk_value_attrs(1, mk_dynamic_tensor_role_fwd());
      DynamicValueAttrs v2_gradient =
          mk_value_attrs(1, mk_dynamic_tensor_role_bwd());

      std::unordered_set<DynamicNodeInvocation> invocation_set = {
          DynamicNodeInvocation{
              /*inputs=*/std::unordered_map<DynamicTensorSlot,
                                            DynamicValueAttrs>{},
              /*node_attrs=*/n1_fwd,
              /*outputs=*/
              std::unordered_map{
                  std::pair{
                      DynamicTensorSlot{
                          /*slot_name=*/TensorSlotName::OUTPUT,
                          /*slot_tensor_role=*/mk_dynamic_tensor_role_fwd(),
                      },
                      v1_activation,
                  },
              },
          },
          DynamicNodeInvocation{
              /*inputs=*/std::unordered_map{
                  std::pair{
                      DynamicTensorSlot{
                          TensorSlotName::INPUT,
                          mk_dynamic_tensor_role_fwd(),
                      },
                      v1_activation,
                  },
              },
              /*node_attrs=*/n2_fwd,
              /*outputs=*/
              std::unordered_map{
                  std::pair{
                      DynamicTensorSlot{
                          TensorSlotName::OUTPUT,
                          mk_dynamic_tensor_role_fwd(),
                      },
                      v2_activation,
                  },
              },
          },
          DynamicNodeInvocation{
              /*inputs=*/std::unordered_map{
                  std::pair{
                      DynamicTensorSlot{
                          TensorSlotName::INPUT,
                          mk_dynamic_tensor_role_fwd(),
                      },
                      v1_activation,
                  },
                  std::pair{
                      DynamicTensorSlot{
                          TensorSlotName::OUTPUT,
                          mk_dynamic_tensor_role_fwd(),
                      },
                      v2_activation,
                  },
                  std::pair{
                      DynamicTensorSlot{
                          TensorSlotName::OUTPUT,
                          mk_dynamic_tensor_role_bwd(),
                      },
                      v2_gradient,
                  },
              },
              /*node_attrs=*/n2_bwd,
              /*outputs=*/
              std::unordered_map{
                  std::pair{
                      DynamicTensorSlot{
                          TensorSlotName::INPUT,
                          mk_dynamic_tensor_role_bwd(),
                      },
                      v1_gradient,
                  },
              },
          },
      };

      return dynamic_open_dataflow_graph_from_invocation_set(invocation_set);
    }();

    ASSERT(get_dynamic_invocation_set(result).size() == 3);
    ASSERT(get_dynamic_invocation_set(result) ==
           get_dynamic_invocation_set(correct));
    ASSERT(dynamic_open_dataflow_graphs_are_isomorphic(result, correct));
  }
}
