#include <doctest/doctest.h>
#include "task-spec/dynamic_graph/dynamic_open_dataflow_graph.h"
#include "task-spec/dynamic_graph/pass_expansion.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("perform_fwd_pass_expansion_for_invocation") {
    auto mk_value_attrs = [](size_t node_id, std::optional<FwbTensorType> const &tensor_type) -> DynamicValueAttrs {
     return DynamicValueAttrs{
        /*pcg_tensor_guid=*/parallel_tensor_guid_t{
          DataflowOutput{
            Node{node_id},
            0_n,
          },
        },
        /*parallel_tensor_shape=*/std::nullopt,
        /*accessor=*/std::nullopt,
        /*tensor_type=*/tensor_type,
      };
    };

    DynamicNodeInvocation invocation = [&]() -> DynamicNodeInvocation {
      DynamicValueAttrs v1 = mk_value_attrs(0, std::nullopt);
      DynamicValueAttrs v2 = mk_value_attrs(1, std::nullopt);
      DynamicValueAttrs v3 = mk_value_attrs(2, std::nullopt);

      return DynamicNodeInvocation{
        /*inputs=*/{v1, v2, v1, v1},
        /*node_attrs=*/DynamicNodeAttrs{
          /*pass_type=*/std::nullopt,
          /*device_coord=*/std::nullopt,
          /*mapping=*/std::nullopt,
        },
        /*outputs=*/{v3},
      };
    }();

    DynamicNodeInvocation result = perform_fwd_pass_expansion_for_invocation(invocation);

    DynamicNodeInvocation correct = [&]() -> DynamicNodeInvocation {
      DynamicValueAttrs v1_fwd = mk_value_attrs(0, FwbTensorType::FORWARD);
      DynamicValueAttrs v2_fwd = mk_value_attrs(1, FwbTensorType::FORWARD);
      DynamicValueAttrs v3_fwd = mk_value_attrs(2, FwbTensorType::FORWARD);

      return DynamicNodeInvocation{
        /*inputs=*/{v1_fwd, v2_fwd, v1_fwd, v1_fwd},
        /*node_attrs=*/DynamicNodeAttrs{
          /*pass_type=*/PassType::FWD,
          /*device_coord=*/std::nullopt,
          /*mapping=*/std::nullopt,
        },
        /*outputs=*/{v3_fwd},
      };
    }();

    ASSERT(result == correct);
  }

  TEST_CASE("perform_bwd_pass_expansion_for_invocation") {
    auto mk_value_attrs = [](size_t node_id, std::optional<FwbTensorType> const &tensor_type) -> DynamicValueAttrs {
     return DynamicValueAttrs{
        /*pcg_tensor_guid=*/parallel_tensor_guid_t{
          DataflowOutput{
            Node{node_id},
            0_n,
          },
        },
        /*parallel_tensor_shape=*/std::nullopt,
        /*accessor=*/std::nullopt,
        /*tensor_type=*/tensor_type,
      };
    };

    DynamicNodeInvocation invocation = [&]() -> DynamicNodeInvocation {
      DynamicValueAttrs v1 = mk_value_attrs(0, std::nullopt);
      DynamicValueAttrs v2 = mk_value_attrs(1, std::nullopt);
      DynamicValueAttrs v3 = mk_value_attrs(2, std::nullopt);

      return DynamicNodeInvocation{
        /*inputs=*/{v1, v2, v1, v1},
        /*node_attrs=*/DynamicNodeAttrs{
          /*pass_type=*/std::nullopt,
          /*device_coord=*/std::nullopt,
          /*mapping=*/std::nullopt,
        },
        /*outputs=*/{v3},
      };
    }();

    DynamicNodeInvocation result = perform_bwd_pass_expansion_for_invocation(invocation);

    DynamicNodeInvocation correct = [&]() -> DynamicNodeInvocation {
      DynamicValueAttrs v1_fwd = mk_value_attrs(0, FwbTensorType::FORWARD);
      DynamicValueAttrs v2_fwd = mk_value_attrs(1, FwbTensorType::FORWARD);
      DynamicValueAttrs v3_fwd = mk_value_attrs(2, FwbTensorType::FORWARD);
      DynamicValueAttrs v1_grad = mk_value_attrs(0, FwbTensorType::GRADIENT);
      DynamicValueAttrs v2_grad = mk_value_attrs(1, FwbTensorType::GRADIENT);
      DynamicValueAttrs v3_grad = mk_value_attrs(2, FwbTensorType::GRADIENT);

      return DynamicNodeInvocation{
        /*inputs=*/{v1_fwd, v2_fwd, v1_fwd, v1_fwd, v3_fwd, v3_grad},
        /*node_attrs=*/DynamicNodeAttrs{
          /*pass_type=*/PassType::BWD,
          /*device_coord=*/std::nullopt,
          /*mapping=*/std::nullopt,
        },
        /*outputs=*/{v1_grad, v2_grad, v1_grad, v1_grad},
      };
    }();

    ASSERT(result == correct);
  }

  TEST_CASE("perform_pass_expansion(DynamicOpenDataflowGraph)") {
    auto mk_node_attrs = [](std::optional<PassType> const &pass_type) -> DynamicNodeAttrs {
      return DynamicNodeAttrs{
        /*pass_type=*/pass_type,
        /*device_coord=*/std::nullopt,
        /*mapping=*/std::nullopt,
      };
    };

    auto mk_value_attrs = [](size_t node_id, std::optional<FwbTensorType> const &tensor_type) -> DynamicValueAttrs {
     return DynamicValueAttrs{
        /*pcg_tensor_guid=*/parallel_tensor_guid_t{
          DataflowOutput{
            Node{node_id},
            0_n,
          },
        },
        /*parallel_tensor_shape=*/std::nullopt,
        /*accessor=*/std::nullopt,
        /*tensor_type=*/tensor_type,
      };
    };

    DynamicOpenDataflowGraph input = [&]() -> DynamicOpenDataflowGraph {

      DynamicNodeAttrs n1 = mk_node_attrs(std::nullopt);
      DynamicNodeAttrs n2 = mk_node_attrs(std::nullopt);

      DynamicValueAttrs v1 = mk_value_attrs(0, std::nullopt);
      DynamicValueAttrs v2 = mk_value_attrs(1, std::nullopt);

      std::unordered_set<DynamicNodeInvocation> invocation_set = {
        DynamicNodeInvocation{
          /*inputs=*/{},
          /*node_attrs=*/n1,
          /*outputs=*/{v1},
        },
        DynamicNodeInvocation{
          /*inputs=*/{v1},
          /*node_attrs=*/n2,
          /*outputs=*/{v2},
        },
      };

      return dynamic_open_dataflow_graph_from_invocation_set(invocation_set);
    }();

    DynamicOpenDataflowGraph result = perform_pass_expansion(input);

    DynamicOpenDataflowGraph correct = [&]() -> DynamicOpenDataflowGraph {
      DynamicNodeAttrs n1_fwd = mk_node_attrs(PassType::FWD);
      DynamicNodeAttrs n2_fwd = mk_node_attrs(PassType::FWD);
      DynamicNodeAttrs n1_bwd = mk_node_attrs(PassType::BWD);
      DynamicNodeAttrs n2_bwd = mk_node_attrs(PassType::BWD);

      DynamicValueAttrs v1_activation = mk_value_attrs(0, FwbTensorType::FORWARD);
      DynamicValueAttrs v1_gradient = mk_value_attrs(0, FwbTensorType::GRADIENT);
      DynamicValueAttrs v2_activation = mk_value_attrs(1, FwbTensorType::FORWARD);
      DynamicValueAttrs v2_gradient = mk_value_attrs(1, FwbTensorType::GRADIENT);

      std::unordered_set<DynamicNodeInvocation> invocation_set = {
        DynamicNodeInvocation{
          /*inputs=*/{},
          /*node_attrs=*/n1_fwd,
          /*outputs=*/{v1_activation},
        },
        DynamicNodeInvocation{
          /*inputs=*/{v1_activation},
          /*node_attrs=*/n2_fwd,
          /*outputs=*/{v2_activation},
        },
        DynamicNodeInvocation{
          /*inputs=*/{v1_activation, v2_activation, v2_gradient},
          /*node_attrs=*/n2_bwd,
          /*outputs=*/{v1_gradient},
        },
      };

      return dynamic_open_dataflow_graph_from_invocation_set(invocation_set);
    }();

    ASSERT(get_dynamic_invocation_set(result).size() == 3);
    ASSERT(get_dynamic_invocation_set(result) == get_dynamic_invocation_set(correct));
    ASSERT(dynamic_open_dataflow_graphs_are_isomorphic(result, correct));
  }
}
