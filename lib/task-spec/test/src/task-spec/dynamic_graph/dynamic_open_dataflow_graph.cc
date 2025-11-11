#include <doctest/doctest.h>
#include "task-spec/dynamic_graph/dynamic_open_dataflow_graph.h"
#include "utils/graph/node/algorithms.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("dynamic_op_dataflow_graph_from_invocation_set") {
    DynamicValueAttrs value_1 = DynamicValueAttrs{
      /*pcg_tensor_guid=*/parallel_tensor_guid_t{
        DataflowOutput{
          Node{1},
          0_n,
        },
      },
      /*parallel_tensor_shape=*/std::nullopt,
      /*accessor=*/std::nullopt,
      /*tensor_type=*/std::nullopt,
    };

    DynamicValueAttrs value_2 = DynamicValueAttrs{
      /*pcg_tensor_guid=*/parallel_tensor_guid_t{
        DataflowOutput{
          Node{2},
          0_n,
        },
      },
      /*parallel_tensor_shape=*/std::nullopt,
      /*accessor=*/std::nullopt,
      /*tensor_type=*/std::nullopt,
    };

    DynamicValueAttrs value_3 = DynamicValueAttrs{
      /*pcg_tensor_guid=*/parallel_tensor_guid_t{
        DataflowOutput{
          Node{3},
          0_n,
        },
      },
      /*parallel_tensor_shape=*/std::nullopt,
      /*accessor=*/std::nullopt,
      /*tensor_type=*/std::nullopt,
    };

    DynamicNodeAttrs node_attrs = DynamicNodeAttrs{
      /*pass_type=*/std::nullopt,
      /*device_coord=*/std::nullopt,
      /*mapping=*/std::nullopt,
    };


    DynamicNodeInvocation invocation_1 = DynamicNodeInvocation{
      /*inputs=*/std::vector{value_1},
      /*node_attrs=*/node_attrs,
      /*outputs=*/std::vector{value_2},
    };

    DynamicNodeInvocation invocation_2 = DynamicNodeInvocation{
      /*inputs=*/std::vector<DynamicValueAttrs>{},
      /*node_attrs=*/node_attrs,
      /*outputs=*/std::vector{value_3},
    };

    DynamicNodeInvocation invocation_3 = DynamicNodeInvocation{
      /*inputs=*/std::vector<DynamicValueAttrs>{value_1, value_2, value_1},
      /*node_attrs=*/node_attrs,
      /*outputs=*/std::vector<DynamicValueAttrs>{},
    };

    std::unordered_set<DynamicNodeInvocation> invocation_set = {
      invocation_1,
      invocation_2,
      invocation_3,
    };

    DynamicOpenDataflowGraph result = dynamic_open_dataflow_graph_from_invocation_set(invocation_set);

    ASSERT(get_nodes(result.raw).size() == 3);
  }
}
