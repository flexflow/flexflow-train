#include "task-spec/dynamic_graph/dynamic_open_dataflow_graph.h"
#include "op-attrs/initializer_attrs.h"
#include "task-spec/dynamic_graph/dynamic_tensor_role.h"
#include "task-spec/dynamic_graph/serializable_dynamic_value_attrs.h"
#include "utils/graph/instances/unordered_set_labelled_open_kwarg_dataflow_graph.h"
#include "utils/graph/node/algorithms.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("dynamic_op_dataflow_graph_from_invocation_set") {

    auto mk_dynamic_value = [](size_t node_id,
                               TensorSlotName slot_name) -> DynamicValueAttrs {
      return DynamicValueAttrs{
          /*tensor_guid=*/dynamic_tensor_guid_t{parallel_tensor_guid_t{
              KwargDataflowOutput{
                  Node{node_id},
                  slot_name,
              },
          }},
          /*parallel_tensor_shape=*/std::nullopt,
          /*shard_coord=*/std::nullopt,
          /*mapping=*/std::nullopt,
          /*accessor=*/std::nullopt,
          /*tensor_type=*/std::nullopt,
      };
    };

    auto mk_slot = [](TensorSlotName slot_name) {
      return DynamicTensorSlot{
          /*slot_name=*/slot_name,
          /*slot_tensor_role=*/std::nullopt,
      };
    };

    DynamicValueAttrs value_1 = mk_dynamic_value(1, TensorSlotName::OUTPUT);
    DynamicValueAttrs value_2 = mk_dynamic_value(2, TensorSlotName::OUTPUT);
    DynamicValueAttrs value_3 = mk_dynamic_value(3, TensorSlotName::OUTPUT);

    DynamicNodeAttrs node_attrs = DynamicNodeAttrs{
        /*task_type=*/std::nullopt,
        /*device_coord=*/std::nullopt,
        /*mapping=*/std::nullopt,
        /*op_attrs=*/std::nullopt,
        /*layer_guid=*/dynamic_layer_guid_t{parallel_layer_guid_t{Node{4}}},
        /*per_device_op_state=*/std::nullopt,
    };

    SUBCASE("correct usage") {
      DynamicNodeInvocation invocation_1 = DynamicNodeInvocation{
          /*inputs=*/std::unordered_map<DynamicTensorSlot, DynamicValueAttrs>{
              {
                  mk_slot(TensorSlotName::INPUT),
                  value_1,
              },
          },
          /*node_attrs=*/node_attrs,
          /*outputs=*/
          std::unordered_map<DynamicTensorSlot, DynamicValueAttrs>{
              {
                  mk_slot(TensorSlotName::OUTPUT),
                  value_2,
              },
          },
      };

      DynamicNodeInvocation invocation_2 = DynamicNodeInvocation{
          /*inputs=*/std::unordered_map<DynamicTensorSlot, DynamicValueAttrs>{},
          /*node_attrs=*/node_attrs,
          /*outputs=*/
          std::unordered_map<DynamicTensorSlot, DynamicValueAttrs>{
              {
                  mk_slot(TensorSlotName::OUTPUT),
                  value_3,
              },
          },
      };

      DynamicNodeInvocation invocation_3 = DynamicNodeInvocation{
          /*inputs=*/std::unordered_map<DynamicTensorSlot, DynamicValueAttrs>{
              {
                  mk_slot(TensorSlotName::INPUT),
                  value_1,
              },
              {
                  mk_slot(TensorSlotName::WEIGHT),
                  value_2,
              },
              {
                  mk_slot(TensorSlotName::BIAS),
                  value_1,
              },
          },
          /*node_attrs=*/node_attrs,
          /*outputs=*/
          std::unordered_map<DynamicTensorSlot, DynamicValueAttrs>{},
      };

      std::unordered_set<DynamicNodeInvocation> invocation_set = {
          invocation_1,
          invocation_2,
          invocation_3,
      };

      DynamicOpenDataflowGraph result =
          dynamic_open_dataflow_graph_from_invocation_set(invocation_set);

      CHECK(dynamic_graph_num_nodes(result) == 3);
    }

    SUBCASE("throws if multiple invocations produce the same value") {
      DynamicNodeInvocation invocation_1 = DynamicNodeInvocation{
          /*inputs=*/std::unordered_map<DynamicTensorSlot, DynamicValueAttrs>{
              {
                  mk_slot(TensorSlotName::INPUT),
                  value_1,
              },
          },
          /*node_attrs=*/node_attrs,
          /*outputs=*/
          std::unordered_map<DynamicTensorSlot, DynamicValueAttrs>{
              {
                  mk_slot(TensorSlotName::OUTPUT),
                  value_2,
              },
          },
      };

      DynamicNodeInvocation invocation_2 = DynamicNodeInvocation{
          /*inputs=*/std::unordered_map<DynamicTensorSlot, DynamicValueAttrs>{},
          /*node_attrs=*/node_attrs,
          /*outputs=*/
          std::unordered_map<DynamicTensorSlot, DynamicValueAttrs>{
              {
                  mk_slot(TensorSlotName::OUTPUT),
                  value_2,
              },
          },
      };

      std::unordered_set<DynamicNodeInvocation> invocation_set = {
          invocation_1,
          invocation_2,
      };

      CHECK_THROWS(
          dynamic_open_dataflow_graph_from_invocation_set(invocation_set));
    }

    SUBCASE("throws if invocations contain/create cycle") {
      DynamicNodeInvocation invocation_1 = DynamicNodeInvocation{
          /*inputs=*/std::unordered_map<DynamicTensorSlot, DynamicValueAttrs>{
              {
                  mk_slot(TensorSlotName::INPUT),
                  value_1,
              },
          },
          /*node_attrs=*/node_attrs,
          /*outputs=*/
          std::unordered_map<DynamicTensorSlot, DynamicValueAttrs>{
              {
                  mk_slot(TensorSlotName::OUTPUT),
                  value_2,
              },
          },
      };

      DynamicNodeInvocation invocation_2 = DynamicNodeInvocation{
          /*inputs=*/std::unordered_map<DynamicTensorSlot, DynamicValueAttrs>{
              {
                  mk_slot(TensorSlotName::INPUT),
                  value_2,
              },
          },
          /*node_attrs=*/node_attrs,
          /*outputs=*/
          std::unordered_map<DynamicTensorSlot, DynamicValueAttrs>{
              {
                  mk_slot(TensorSlotName::OUTPUT),
                  value_1,
              },
          },
      };

      std::unordered_set<DynamicNodeInvocation> invocation_set = {
          invocation_1,
          invocation_2,
      };

      CHECK_THROWS(
          dynamic_open_dataflow_graph_from_invocation_set(invocation_set));
    }
  }

  TEST_CASE("get_dynamic_slot_sites") {
    auto mk_dynamic_value = [](int node_id,
                               TensorSlotName slot_name) -> DynamicValueAttrs {
      return DynamicValueAttrs{
          /*tensor_guid=*/dynamic_tensor_guid_t{parallel_tensor_guid_t{
              KwargDataflowOutput{
                  Node{static_cast<size_t>(node_id)},
                  slot_name,
              },
          }},
          /*parallel_tensor_shape=*/std::nullopt,
          /*shard_coord=*/std::nullopt,
          /*mapping=*/std::nullopt,
          /*accessor=*/std::nullopt,
          /*tensor_type=*/std::nullopt,
      };
    };

    DynamicValueAttrs value_1 = mk_dynamic_value(1, TensorSlotName::OUTPUT);
    DynamicValueAttrs value_2 = mk_dynamic_value(2, TensorSlotName::OUTPUT);
    DynamicValueAttrs value_3 = mk_dynamic_value(3, TensorSlotName::OUTPUT);

    DynamicNodeAttrs node_attrs = DynamicNodeAttrs{
        /*task_type=*/std::nullopt,
        /*device_coord=*/std::nullopt,
        /*mapping=*/std::nullopt,
        /*op_attrs=*/std::nullopt,
        /*layer_guid=*/dynamic_layer_guid_t{parallel_layer_guid_t{Node{4}}},
        /*per_device_op_state=*/std::nullopt,
    };

    DynamicNodeInvocation invocation_1 = DynamicNodeInvocation{
        /*inputs=*/std::unordered_map<DynamicTensorSlot, DynamicValueAttrs>{
            {
                DynamicTensorSlot{
                    /*slot_name=*/TensorSlotName::INPUT,
                    /*slot_tensor_role=*/std::nullopt,
                },
                value_1,
            },
        },
        /*node_attrs=*/node_attrs,
        /*outputs=*/
        std::unordered_map<DynamicTensorSlot, DynamicValueAttrs>{
            {
                DynamicTensorSlot{
                    /*slot_name=*/TensorSlotName::OUTPUT,
                    /*slot_tensor_role=*/std::nullopt,
                },
                value_2,
            },
        },
    };

    DynamicNodeInvocation invocation_2 = DynamicNodeInvocation{
        /*inputs=*/std::unordered_map<DynamicTensorSlot, DynamicValueAttrs>{
            {
                DynamicTensorSlot{
                    /*slot_name=*/TensorSlotName::INPUT,
                    /*slot_tensor_role=*/std::nullopt,
                },
                value_2,
            },
            {
                DynamicTensorSlot{
                    /*slot_name=*/TensorSlotName::WEIGHT,
                    /*slot_tensor_role=*/std::nullopt,
                },
                value_3,
            },
        },
        /*node_attrs=*/node_attrs,
        /*outputs=*/{},
    };

    DynamicOpenDataflowGraph g =
        dynamic_open_dataflow_graph_from_invocation_set(
            std::unordered_set{invocation_1, invocation_2});

    std::unordered_set<DynamicSlotSite> result = get_dynamic_slot_sites(g);

    auto mk_internal_slot_site = [](DynamicNodeInvocation const &invocation,
                                    TensorDirection direction,
                                    TensorSlotName slot_name) {
      return DynamicSlotSite{
          InternalDynamicSlotSite{
              /*invocation=*/invocation,
              /*direction=*/direction,
              /*slot_name=*/
              DynamicTensorSlot{
                  /*slot_name=*/slot_name,
                  /*slot_tensor_role=*/std::nullopt,
              },
          },
      };
    };

    std::unordered_set<DynamicSlotSite> correct = {
        DynamicSlotSite{
            ExternalDynamicSlotSite{
                value_1,
            },
        },
        DynamicSlotSite{
            ExternalDynamicSlotSite{
                value_3,
            },
        },
        mk_internal_slot_site(
            invocation_1, TensorDirection::INCOMING, TensorSlotName::INPUT),
        mk_internal_slot_site(
            invocation_1, TensorDirection::OUTPUT, TensorSlotName::OUTPUT),
        mk_internal_slot_site(
            invocation_2, TensorDirection::INCOMING, TensorSlotName::INPUT),
        mk_internal_slot_site(
            invocation_2, TensorDirection::INCOMING, TensorSlotName::WEIGHT),
    };

    CHECK(result == correct);
  }

  TEST_CASE(
      "labelled_open_kwarg_dataflow_graph_from_dynamic_open_dataflow_graph") {
    dynamic_layer_guid_t layer_guid = dynamic_layer_guid_t{
        parallel_layer_guid_t{
            Node{0},
        },
    };

    dynamic_tensor_guid_t tensor_guid = dynamic_tensor_guid_t{
        parallel_tensor_guid_t{
            KwargDataflowOutput<TensorSlotName>{
                /*node=*/Node{0},
                /*slot_name=*/TensorSlotName::OUTPUT,
            },
        },
    };

    TrainingOperationAttrs weight_attrs = TrainingOperationAttrs{
        PCGOperatorAttrs{
            WeightAttrs{
                /*tensor_shape=*/TensorShape{
                    /*dims=*/TensorDims{
                        FFOrdered{
                            4_p,
                            3_p,
                        },
                    },
                    /*data_type=*/DataType::FLOAT,
                },
                /*initializer=*/make_zero_initializer(),
            },
        },
    };

    DynamicNodeAttrs fwd_weight_node_attrs = DynamicNodeAttrs{
        /*task_type=*/DynamicTaskType::FWD,
        /*device_coord=*/std::nullopt,
        /*mapping=*/std::nullopt,
        /*op_attrs=*/weight_attrs,
        /*layer_guid=*/layer_guid,
        /*per_device_op_state=*/std::nullopt,
    };

    DynamicTensorSlot fwd_weight_output_slot1 = DynamicTensorSlot{
        /*slot_name=*/TensorSlotName::OUTPUT,
        /*slot_tensor_role=*/mk_dynamic_tensor_role_fwd(),
    };

    DynamicValueAttrs fwd_weight_output_attrs1 = DynamicValueAttrs{
        /*tensor_guid=*/tensor_guid,
        /*parallel_tensor_shape=*/std::nullopt,
        /*shard_coord=*/std::nullopt,
        /*mapping=*/std::nullopt,
        /*accessor=*/std::nullopt,
        /*role=*/mk_dynamic_tensor_role_fwd(),
    };

    DynamicNodeInvocation weight_invocation = DynamicNodeInvocation{
        /*inputs=*/{},
        /*node_attrs=*/fwd_weight_node_attrs,
        /*outputs=*/
        std::unordered_map<DynamicTensorSlot, DynamicValueAttrs>{
            {
                fwd_weight_output_slot1,
                fwd_weight_output_attrs1,
            },
        },
    };

    DynamicNodeAttrs upd_weight_node_attrs = DynamicNodeAttrs{
        /*task_type=*/DynamicTaskType::UPD,
        /*device_coord=*/std::nullopt,
        /*mapping=*/std::nullopt,
        /*op_attrs=*/weight_attrs,
        /*layer_guid=*/layer_guid,
        /*per_device_op_state=*/std::nullopt,
    };

    DynamicTensorSlot upd_weight_input_slot2 = DynamicTensorSlot{
        /*slot_name=*/TensorSlotName::OUTPUT,
        /*slot_tensor_role=*/mk_dynamic_tensor_role_bwd(),
    };

    DynamicValueAttrs upd_weight_input_attrs2 = DynamicValueAttrs{
        /*tensor_guid=*/tensor_guid,
        /*parallel_tensor_shape=*/std::nullopt,
        /*shard_coord=*/std::nullopt,
        /*mapping=*/std::nullopt,
        /*accessor=*/std::nullopt,
        /*role=*/mk_dynamic_tensor_role_bwd(),
    };

    DynamicTensorSlot upd_weight_input_slot3 = DynamicTensorSlot{
        /*slot_name=*/TensorSlotName::OUTPUT,
        /*slot_tensor_role=*/
        mk_dynamic_tensor_role_opt(OptimizerSlotName::SGD_V),
    };

    DynamicValueAttrs upd_weight_input_attrs3 = DynamicValueAttrs{
        /*tensor_guid=*/tensor_guid,
        /*parallel_tensor_shape=*/std::nullopt,
        /*shard_coord=*/std::nullopt,
        /*mapping=*/std::nullopt,
        /*accessor=*/std::nullopt,
        /*role=*/mk_dynamic_tensor_role_opt(OptimizerSlotName::SGD_V),
    };

    DynamicOpenDataflowGraph input =
        dynamic_open_dataflow_graph_from_invocation_set(
            /*invocations=*/{
                weight_invocation,
                DynamicNodeInvocation{/*inputs=*/{
                                          {
                                              fwd_weight_output_slot1,
                                              fwd_weight_output_attrs1,
                                          },
                                          {
                                              upd_weight_input_slot2,
                                              upd_weight_input_attrs2,
                                          },
                                          {
                                              upd_weight_input_slot3,
                                              upd_weight_input_attrs3,
                                          },
                                      },
                                      /*node_attrs=*/upd_weight_node_attrs,
                                      /*outputs=*/{}},
            });

    std::pair<LabelledOpenKwargDataflowGraph<DynamicNodeAttrs,
                                             DynamicValueAttrs,
                                             int,
                                             DynamicTensorSlot>,
              bidict<Node, DynamicNodeInvocation>>
        result =
            labelled_open_kwarg_dataflow_graph_from_dynamic_open_dataflow_graph(
                input);

    LabelledOpenKwargDataflowGraph<DynamicNodeAttrs,
                                   DynamicValueAttrs,
                                   int,
                                   DynamicTensorSlot>
        correct = LabelledOpenKwargDataflowGraph<DynamicNodeAttrs,
                                                 DynamicValueAttrs,
                                                 int,
                                                 DynamicTensorSlot>::
            create<UnorderedSetLabelledOpenKwargDataflowGraph<
                DynamicNodeAttrs,
                DynamicValueAttrs,
                int,
                DynamicTensorSlot>>();

    KwargNodeAddedResult<DynamicTensorSlot> fwd_weight_added = correct.add_node(
        /*node_label=*/fwd_weight_node_attrs,
        /*inputs=*/{},
        /*output_labels=*/
        {
            {
                fwd_weight_output_slot1,
                fwd_weight_output_attrs1,
            },
        });

    KwargNodeAddedResult<DynamicTensorSlot> upd_weight_added = correct.add_node(
        /*node_label=*/fwd_weight_node_attrs,
        /*inputs=*/{},
        /*output_labels=*/
        {
            {
                fwd_weight_output_slot1,
                fwd_weight_output_attrs1,
            },
        });
  }
}
