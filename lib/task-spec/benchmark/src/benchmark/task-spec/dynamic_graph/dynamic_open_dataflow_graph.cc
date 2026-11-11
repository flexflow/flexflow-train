#include "benchmark/task-spec/dynamic_graph/dynamic_open_dataflow_graph.h"
#include "op-attrs/ops/element_unary.h"
#include "task-spec/dynamic_graph/dynamic_node_invocation.dtg.h"
#include "task-spec/dynamic_graph/dynamic_open_dataflow_graph.h"
#include "task-spec/dynamic_graph/dynamic_value_attrs.dtg.h"
#include "utils/benchmark_utils/loop.h"
#include "utils/containers/at_idx.h"
#include "utils/containers/get_only.h"
#include "utils/containers/range.h"
#include "utils/containers/range_inclusive.h"
#include "utils/nonnegative_int/nonnegative_int.h"
#include <vector>

namespace FlexFlow {

static std::set<DynamicNodeInvocation> create_benchmark_invocation_set() {
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
        /*create_grad=*/std::nullopt,
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
        /*task_shard=*/std::nullopt,
    };
  };

  auto mk_node = [](size_t node_id) {
    return DynamicNodeAttrs{
        /*task_type=*/std::nullopt,
        /*device_coord=*/std::nullopt,
        /*mapping=*/std::nullopt,
        /*op_attrs=*/
        TrainingOperationAttrs{PCGOperatorAttrs{make_relu_attrs()}},
        /*layer_guid=*/
        dynamic_layer_guid_t{parallel_layer_guid_t{Node{node_id}}},
        /*per_device_op_state=*/std::nullopt,
    };
  };

  constexpr int num_invocations = 1000;
  std::vector<DynamicValueAttrs> values =
      transform(range_inclusive(num_invocations), [&](int idx) {
        return mk_dynamic_value(idx, TensorSlotName::OUTPUT);
      });

  std::set<DynamicNodeInvocation> result =
      set_of(transform(range(num_invocations), [&](int idx) {
        return DynamicNodeInvocation{
            /*inputs=*/std::map<DynamicTensorSlot, DynamicValueAttrs>{
                {
                    mk_slot(TensorSlotName::INPUT),
                    values.at(idx),
                },
            },
            /*node_attrs=*/mk_node(idx),
            /*outputs=*/
            std::map<DynamicTensorSlot, DynamicValueAttrs>{
                {
                    mk_slot(TensorSlotName::OUTPUT),
                    values.at(idx + 1),
                },
            },
        };
      }));
  return result;
}

void benchmark_make_empty_dynamic_open_dataflow_graph(bool dry_run) {
  LOOP(10000, dry_run) {
    make_empty_dynamic_open_dataflow_graph();
  }
}

void benchmark_dynamic_open_dataflow_graph_from_invocation_set(bool dry_run) {
  std::set<DynamicNodeInvocation> invocation_set =
      create_benchmark_invocation_set();
  LOOP(10, dry_run) {
    dynamic_open_dataflow_graph_from_invocation_set(invocation_set);
  }
}

void benchmark_dynamic_graph_get_id_for_invocation(bool dry_run) {
  std::set<DynamicNodeInvocation> invocation_set =
      create_benchmark_invocation_set();
  DynamicOpenDataflowGraph dg =
      dynamic_open_dataflow_graph_from_invocation_set(invocation_set);
  DynamicNodeInvocation invocation =
      at_idx(invocation_set, nonnegative_int{invocation_set.size() / 2});
  LOOP(1000, dry_run) {
    dynamic_graph_get_id_for_invocation(dg, invocation);
  }
}

void benchmark_dynamic_graph_get_invocation_for_id(bool dry_run) {
  std::set<DynamicNodeInvocation> invocation_set =
      create_benchmark_invocation_set();
  DynamicOpenDataflowGraph dg =
      dynamic_open_dataflow_graph_from_invocation_set(invocation_set);
  DynamicNodeInvocation invocation =
      at_idx(invocation_set, nonnegative_int{invocation_set.size() / 2});
  dynamic_invocation_id_t id =
      dynamic_graph_get_id_for_invocation(dg, invocation);
  LOOP(1000, dry_run) {
    dynamic_graph_get_invocation_for_id(dg, id);
  }
}

void benchmark_dynamic_graph_get_id_for_value(bool dry_run) {
  std::set<DynamicNodeInvocation> invocation_set =
      create_benchmark_invocation_set();
  DynamicOpenDataflowGraph dg =
      dynamic_open_dataflow_graph_from_invocation_set(invocation_set);
  DynamicNodeInvocation invocation =
      at_idx(invocation_set, nonnegative_int{invocation_set.size() / 2});
  DynamicValueAttrs value = get_only(invocation.outputs).second;
  LOOP(1000, dry_run) {
    dynamic_graph_get_id_for_value(dg, value);
  }
}

void benchmark_dynamic_graph_get_value_for_id(bool dry_run) {
  std::set<DynamicNodeInvocation> invocation_set =
      create_benchmark_invocation_set();
  DynamicOpenDataflowGraph dg =
      dynamic_open_dataflow_graph_from_invocation_set(invocation_set);
  DynamicNodeInvocation invocation =
      at_idx(invocation_set, nonnegative_int{invocation_set.size() / 2});
  DynamicValueAttrs value = get_only(invocation.outputs).second;
  dynamic_value_id_t id = dynamic_graph_get_id_for_value(dg, value);
  LOOP(1000, dry_run) {
    dynamic_graph_get_value_for_id(dg, id);
  }
}

} // namespace FlexFlow
