#include "task-spec/dynamic_graph/pass_expansion.h"
#include "op-attrs/ops/element_unary.h"
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
          /*tensor_guid=*/dynamic_tensor_guid_t{parallel_tensor_guid_t{
              KwargDataflowOutput<TensorSlotName>{
                  Node{node_id},
                  TensorSlotName::OUTPUT,
              },
          }},
          /*parallel_tensor_shape=*/std::nullopt,
          /*shard_coord=*/std::nullopt,
          /*mapping=*/std::nullopt,
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
          /*task_shard=*/std::nullopt,
      };
    };

    dynamic_layer_guid_t layer_guid{parallel_layer_guid_t{Node{20}}};

    TrainingOperationAttrs op_attrs = TrainingOperationAttrs{
        PCGOperatorAttrs{
            LinearAttrs{
                /*out_channels=*/8_p,
                /*use_bias=*/true,
                /*data_type=*/DataType::FLOAT,
                /*activation=*/std::nullopt,
                /*regularizer=*/std::nullopt,
            },
        },
    };

    DynamicNodeInvocation invocation = [&]() -> DynamicNodeInvocation {
      DynamicValueAttrs v1 = mk_value_attrs(0, std::nullopt);
      DynamicValueAttrs v2 = mk_value_attrs(1, std::nullopt);
      DynamicValueAttrs v3 = mk_value_attrs(2, std::nullopt);

      return DynamicNodeInvocation{
          /*inputs=*/{
              {mk_slot(TensorSlotName::INPUT, std::nullopt), v1},
              {mk_slot(TensorSlotName::WEIGHT, std::nullopt), v2},
              {mk_slot(TensorSlotName::BIAS, std::nullopt), v1},
          },
          /*node_attrs=*/
          DynamicNodeAttrs{
              /*task_type=*/std::nullopt,
              /*device_coord=*/std::nullopt,
              /*mapping=*/std::nullopt,
              /*op_attrs=*/op_attrs,
              /*layer_guid=*/layer_guid,
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
          },
          /*node_attrs=*/
          DynamicNodeAttrs{
              /*task_type=*/DynamicTaskType::FWD,
              /*device_coord=*/std::nullopt,
              /*mapping=*/std::nullopt,
              /*op_attrs=*/op_attrs,
              /*layer_guid=*/layer_guid,
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
          /*tensor_guid=*/dynamic_tensor_guid_t{parallel_tensor_guid_t{
              KwargDataflowOutput<TensorSlotName>{
                  Node{node_id},
                  TensorSlotName::OUTPUT,
              },
          }},
          /*parallel_tensor_shape=*/std::nullopt,
          /*shard_coord=*/std::nullopt,
          /*mapping=*/std::nullopt,
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
          /*task_shard=*/std::nullopt,
      };
    };

    dynamic_layer_guid_t layer_guid{parallel_layer_guid_t{Node{20}}};

    DynamicValueAttrs v1 = mk_value_attrs(0, std::nullopt);
    DynamicValueAttrs v2 = mk_value_attrs(1, std::nullopt);
    DynamicValueAttrs v3 = mk_value_attrs(2, std::nullopt);

    DynamicTensorRole fwd_role = DynamicTensorRole{FwbTensorType::FORWARD};
    DynamicTensorRole grad_role = DynamicTensorRole{FwbTensorType::GRADIENT};

    DynamicValueAttrs v1_fwd = mk_value_attrs(0, fwd_role);
    DynamicValueAttrs v2_fwd = mk_value_attrs(1, fwd_role);
    DynamicValueAttrs v3_fwd = mk_value_attrs(2, fwd_role);
    DynamicValueAttrs v1_grad = mk_value_attrs(0, grad_role);
    DynamicValueAttrs v2_grad = mk_value_attrs(1, grad_role);
    DynamicValueAttrs v3_grad = mk_value_attrs(2, grad_role);

    SUBCASE("normal operator") {
      TrainingOperationAttrs op_attrs = TrainingOperationAttrs{
          PCGOperatorAttrs{
              LinearAttrs{
                  /*out_channels=*/8_p,
                  /*use_bias=*/true,
                  /*data_type=*/DataType::FLOAT,
                  /*activation=*/std::nullopt,
                  /*regularizer=*/std::nullopt,
              },
          },
      };

      DynamicNodeInvocation invocation = [&]() -> DynamicNodeInvocation {
        return DynamicNodeInvocation{
            /*inputs=*/{
                {mk_slot(TensorSlotName::INPUT, std::nullopt), v1},
                {mk_slot(TensorSlotName::WEIGHT, std::nullopt), v2},
                {mk_slot(TensorSlotName::BIAS, std::nullopt), v1},
            },
            /*node_attrs=*/
            DynamicNodeAttrs{
                /*task_type=*/std::nullopt,
                /*device_coord=*/std::nullopt,
                /*mapping=*/std::nullopt,
                /*op_attrs=*/op_attrs,
                /*layer_guid=*/layer_guid,
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
        return DynamicNodeInvocation{
            /*inputs=*/{
                {mk_slot(TensorSlotName::INPUT, fwd_role), v1_fwd},
                {mk_slot(TensorSlotName::WEIGHT, fwd_role), v2_fwd},
                {mk_slot(TensorSlotName::BIAS, fwd_role), v1_fwd},
                {mk_slot(TensorSlotName::OUTPUT, fwd_role), v3_fwd},
                {mk_slot(TensorSlotName::OUTPUT, grad_role), v3_grad},
            },
            /*node_attrs=*/
            DynamicNodeAttrs{
                /*pass_type=*/DynamicTaskType::BWD,
                /*device_coord=*/std::nullopt,
                /*mapping=*/std::nullopt,
                /*op_attrs=*/op_attrs,
                /*layer_guid=*/layer_guid,
                /*per_device_op_state=*/std::nullopt,
            },
            /*outputs=*/
            {
                {mk_slot(TensorSlotName::INPUT, grad_role), v1_grad},
                {mk_slot(TensorSlotName::WEIGHT, grad_role), v2_grad},
                {mk_slot(TensorSlotName::BIAS, grad_role), v1_grad},
            },
        };
      }();

      ASSERT(result == correct);
    }

    SUBCASE("replicate operator optimization") {
      TrainingOperationAttrs op_attrs = TrainingOperationAttrs{
          PCGOperatorAttrs{
              ReplicateAttrs{
                  /*replicate_degree=*/2_p,
              },
          },
      };

      DynamicNodeInvocation invocation = [&]() -> DynamicNodeInvocation {
        return DynamicNodeInvocation{
            /*inputs=*/{
                {mk_slot(TensorSlotName::INPUT, std::nullopt), v1},
            },
            /*node_attrs=*/
            DynamicNodeAttrs{
                /*task_type=*/std::nullopt,
                /*device_coord=*/std::nullopt,
                /*mapping=*/std::nullopt,
                /*op_attrs=*/op_attrs,
                /*layer_guid=*/layer_guid,
                /*per_device_op_state=*/std::nullopt,
            },
            /*outputs=*/
            {
                {mk_slot(TensorSlotName::OUTPUT, std::nullopt), v2},
            },
        };
      }();

      DynamicNodeInvocation result =
          perform_bwd_pass_expansion_for_invocation(invocation);

      DynamicNodeInvocation correct = [&]() -> DynamicNodeInvocation {
        DynamicTensorRole fwd_role = DynamicTensorRole{FwbTensorType::FORWARD};
        DynamicTensorRole grad_role =
            DynamicTensorRole{FwbTensorType::GRADIENT};

        return DynamicNodeInvocation{
            /*inputs=*/{
                {mk_slot(TensorSlotName::OUTPUT, fwd_role), v2_fwd},
                {mk_slot(TensorSlotName::OUTPUT, grad_role), v2_grad},
            },
            /*node_attrs=*/
            DynamicNodeAttrs{
                /*pass_type=*/DynamicTaskType::BWD,
                /*device_coord=*/std::nullopt,
                /*mapping=*/std::nullopt,
                /*op_attrs=*/op_attrs,
                /*layer_guid=*/layer_guid,
                /*per_device_op_state=*/std::nullopt,
            },
            /*outputs=*/
            {
                {mk_slot(TensorSlotName::INPUT, grad_role), v1_grad},
            },
        };
      }();

      ASSERT(result == correct);
    }
  }

  TEST_CASE("perform_pass_expansion(DynamicOpenDataflowGraph)") {
    auto mk_node_attrs = [](size_t layer_id,
                            TrainingOperationAttrs const &op_attrs,
                            std::optional<DynamicTaskType> const &pass_type)
        -> DynamicNodeAttrs {
      return DynamicNodeAttrs{
          /*pass_type=*/pass_type,
          /*device_coord=*/std::nullopt,
          /*mapping=*/std::nullopt,
          /*op_attrs=*/op_attrs,
          /*layer_guid=*/
          dynamic_layer_guid_t{parallel_layer_guid_t{Node{layer_id}}},
          /*per_device_op_state=*/std::nullopt,
      };
    };

    auto mk_value_attrs =
        [](size_t node_id, std::optional<DynamicTensorRole> const &tensor_type)
        -> DynamicValueAttrs {
      return DynamicValueAttrs{
          /*tensor_guid=*/dynamic_tensor_guid_t{parallel_tensor_guid_t{
              KwargDataflowOutput{
                  Node{node_id},
                  TensorSlotName::OUTPUT,
              },
          }},
          /*parallel_tensor_shape=*/std::nullopt,
          /*shard_coord=*/std::nullopt,
          /*mapping=*/std::nullopt,
          /*accessor=*/std::nullopt,
          /*role=*/tensor_type,
      };
    };

    TrainingOperationAttrs input_op_attrs = TrainingOperationAttrs{
        PCGOperatorAttrs{
            InputAttrs{
                TensorShape{
                    TensorDims{
                        FFOrdered<positive_int>{
                            4_p,
                            8_p,
                        },
                    },
                    DataType::FLOAT,
                },
            },
        },
    };

    TrainingOperationAttrs relu_op_attrs = TrainingOperationAttrs{
        PCGOperatorAttrs{
            make_relu_attrs(),
        },
    };

    DynamicOpenDataflowGraph input = [&]() -> DynamicOpenDataflowGraph {
      DynamicNodeAttrs n1 = mk_node_attrs(10, input_op_attrs, std::nullopt);
      DynamicNodeAttrs n2 = mk_node_attrs(11, relu_op_attrs, std::nullopt);

      DynamicValueAttrs v1 = mk_value_attrs(0, std::nullopt);
      DynamicValueAttrs v2 = mk_value_attrs(1, std::nullopt);

      std::unordered_set<DynamicNodeInvocation> invocation_set = {
          DynamicNodeInvocation{
              /*inputs=*/std::map<DynamicTensorSlot, DynamicValueAttrs>{},
              /*node_attrs=*/n1,
              /*outputs=*/
              std::map<DynamicTensorSlot, DynamicValueAttrs>{
                  {
                      DynamicTensorSlot{
                          /*slot_name=*/TensorSlotName::OUTPUT,
                          /*slot_tensor_role=*/std::nullopt,
                          /*task_shard=*/std::nullopt,
                      },
                      v1,
                  },
              },
          },
          DynamicNodeInvocation{
              /*inputs=*/std::map<DynamicTensorSlot, DynamicValueAttrs>{
                  {
                    DynamicTensorSlot{
                       /*slot_name=*/TensorSlotName::INPUT,
                       /*slot_tensor_role=*/std::nullopt,
                       /*task_shard=*/std::nullopt,
                    },
                    v1,
                  },
              },
              /*node_attrs=*/n2,
              /*outputs=*/
              std::map<DynamicTensorSlot, DynamicValueAttrs>{
                  {
                    DynamicTensorSlot{
                      /*slot_name=*/TensorSlotName::OUTPUT,
                      /*slot_tensor_role=*/std::nullopt,
                      /*task_shard=*/std::nullopt,
                    },
                    v2,
                  },
              },
          },
      };

      return dynamic_open_dataflow_graph_from_invocation_set(invocation_set);
    }();

    DynamicOpenDataflowGraph result = perform_pass_expansion(input);

    DynamicOpenDataflowGraph correct = [&]() -> DynamicOpenDataflowGraph {
      DynamicNodeAttrs n1_fwd =
          mk_node_attrs(10, input_op_attrs, DynamicTaskType::FWD);
      DynamicNodeAttrs n2_fwd =
          mk_node_attrs(11, relu_op_attrs, DynamicTaskType::FWD);
      DynamicNodeAttrs n1_bwd =
          mk_node_attrs(10, input_op_attrs, DynamicTaskType::BWD);
      DynamicNodeAttrs n2_bwd =
          mk_node_attrs(11, relu_op_attrs, DynamicTaskType::BWD);

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
              /*inputs=*/std::map<DynamicTensorSlot, DynamicValueAttrs>{},
              /*node_attrs=*/n1_fwd,
              /*outputs=*/
              std::map{
                  std::pair{
                      DynamicTensorSlot{
                          /*slot_name=*/TensorSlotName::OUTPUT,
                          /*slot_tensor_role=*/mk_dynamic_tensor_role_fwd(),
                          /*task_shard=*/std::nullopt,
                      },
                      v1_activation,
                  },
              },
          },
          DynamicNodeInvocation{
              /*inputs=*/std::map{
                  std::pair{
                      DynamicTensorSlot{
                          TensorSlotName::INPUT,
                          mk_dynamic_tensor_role_fwd(),
                          /*task_shard=*/std::nullopt,
                      },
                      v1_activation,
                  },
              },
              /*node_attrs=*/n2_fwd,
              /*outputs=*/
              std::map{
                  std::pair{
                      DynamicTensorSlot{
                          TensorSlotName::OUTPUT,
                          mk_dynamic_tensor_role_fwd(),
                          /*task_shard=*/std::nullopt,
                      },
                      v2_activation,
                  },
              },
          },
          DynamicNodeInvocation{
              /*inputs=*/std::map{
                  std::pair{
                      DynamicTensorSlot{
                          TensorSlotName::INPUT,
                          mk_dynamic_tensor_role_fwd(),
                          /*task_shard=*/std::nullopt,
                      },
                      v1_activation,
                  },
                  std::pair{
                      DynamicTensorSlot{
                          TensorSlotName::OUTPUT,
                          mk_dynamic_tensor_role_fwd(),
                          /*task_shard=*/std::nullopt,
                      },
                      v2_activation,
                  },
                  std::pair{
                      DynamicTensorSlot{
                          TensorSlotName::OUTPUT,
                          mk_dynamic_tensor_role_bwd(),
                          /*task_shard=*/std::nullopt,
                      },
                      v2_gradient,
                  },
              },
              /*node_attrs=*/n2_bwd,
              /*outputs=*/
              std::map{
                  std::pair{
                      DynamicTensorSlot{
                          TensorSlotName::INPUT,
                          mk_dynamic_tensor_role_bwd(),
                          /*task_shard=*/std::nullopt,
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
