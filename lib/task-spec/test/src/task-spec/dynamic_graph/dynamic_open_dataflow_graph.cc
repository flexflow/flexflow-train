#include <doctest/doctest.h>
#include "task-spec/dynamic_graph/dynamic_open_dataflow_graph.h"
#include "utils/graph/node/algorithms.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("dynamic_op_dataflow_graph_from_invocation_set") {
    DynamicValueAttrs value_1 = DynamicValueAttrs{
      /*pcg_tensor_guid=*/parallel_tensor_guid_t{
        KwargDataflowOutput{
          Node{1},
          TensorSlotName::OUTPUT,
        },
      },
      /*parallel_tensor_shape=*/std::nullopt,
      /*shard_coord=*/std::nullopt,
      /*accessor=*/std::nullopt,
      /*tensor_type=*/std::nullopt,
    };

    DynamicValueAttrs value_2 = DynamicValueAttrs{
      /*pcg_tensor_guid=*/parallel_tensor_guid_t{
        KwargDataflowOutput{
          Node{2},
          TensorSlotName::OUTPUT,
        },
      },
      /*parallel_tensor_shape=*/std::nullopt,
      /*shard_coord=*/std::nullopt,
      /*accessor=*/std::nullopt,
      /*tensor_type=*/std::nullopt,
    };

    DynamicValueAttrs value_3 = DynamicValueAttrs{
      /*pcg_tensor_guid=*/parallel_tensor_guid_t{
        KwargDataflowOutput{
          Node{3},
          TensorSlotName::OUTPUT,
        },
      },
      /*parallel_tensor_shape=*/std::nullopt,
      /*shard_coord=*/std::nullopt,
      /*accessor=*/std::nullopt,
      /*tensor_type=*/std::nullopt,
    };

    DynamicNodeAttrs node_attrs = DynamicNodeAttrs{
      /*task_type=*/std::nullopt,
      /*device_coord=*/std::nullopt,
      /*mapping=*/std::nullopt,
      /*op_attrs=*/std::nullopt,
      /*pcg_layer_guid=*/parallel_layer_guid_t{Node{4}},
      /*per_device_op_state=*/std::nullopt,
    };


    DynamicNodeInvocation invocation_1 = DynamicNodeInvocation{
      /*inputs=*/std::unordered_map<DynamicTensorSlot, DynamicValueAttrs>{
        {
          DynamicTensorSlot{
            /*slot_name=*/TensorSlotName::INPUT,
            /*slot_tensor_role=*/std::nullopt,
          },
          value_1
        },
      },
      /*node_attrs=*/node_attrs,
      /*outputs=*/std::unordered_map<DynamicTensorSlot, DynamicValueAttrs>{
        {
          DynamicTensorSlot{
            /*slot_name=*/TensorSlotName::OUTPUT,
            /*slot_tensor_role=*/std::nullopt,
          },
          value_2
        },
      },
    };

    DynamicNodeInvocation invocation_2 = DynamicNodeInvocation{
      /*inputs=*/std::unordered_map<DynamicTensorSlot, DynamicValueAttrs>{},
      /*node_attrs=*/node_attrs,
      /*outputs=*/std::unordered_map<DynamicTensorSlot, DynamicValueAttrs>{
        {
          DynamicTensorSlot{
            /*slot_name=*/TensorSlotName::OUTPUT,
            /*slot_tensor_role=*/std::nullopt,
          },
          value_3,
        },
      },
    };

    DynamicNodeInvocation invocation_3 = DynamicNodeInvocation{
      /*inputs=*/std::unordered_map<DynamicTensorSlot, DynamicValueAttrs>{
        {
          DynamicTensorSlot{
            /*slot_name=*/TensorSlotName::INPUT,
            /*slot_tensor_role=*/std::nullopt,
          },
          value_1, 
        },
        {
          DynamicTensorSlot{
            /*slot_name=*/TensorSlotName::WEIGHT,
            /*slot_tensor_role=*/std::nullopt,
          },
          value_2, 
        },
        {
          DynamicTensorSlot{
            /*slot_name=*/TensorSlotName::BIAS,
            /*slot_tensor_role=*/std::nullopt,
          },
          value_1,
        },
      },
      /*node_attrs=*/node_attrs,
      /*outputs=*/std::unordered_map<DynamicTensorSlot, DynamicValueAttrs>{},
    };

    std::unordered_set<DynamicNodeInvocation> invocation_set = {
      invocation_1,
      invocation_2,
      invocation_3,
    };

    DynamicOpenDataflowGraph result = dynamic_open_dataflow_graph_from_invocation_set(invocation_set);

    ASSERT(dynamic_graph_num_nodes(result) == 3);
  }
}
