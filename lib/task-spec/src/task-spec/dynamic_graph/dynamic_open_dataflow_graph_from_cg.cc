#include "task-spec/dynamic_graph/dynamic_open_dataflow_graph_from_cg.h"
#include "op-attrs/pcg_operator_attrs.h"
#include "pcg/computation_graph.h"
#include "task-spec/dynamic_graph/dynamic_layer_guid_t.dtg.h"
#include "task-spec/dynamic_graph/dynamic_open_dataflow_graph.h"
#include "task-spec/dynamic_graph/dynamic_tensor_role.h"
#include <optional>
#include <unordered_map>

namespace FlexFlow {

DynamicOpenDataflowGraph
    make_dynamic_open_dataflow_graph_from_cg(ComputationGraph const &cg) {
  DynamicOpenDataflowGraph result = make_empty_dynamic_open_dataflow_graph();

  for (auto const &[layer, attrs] : get_layer_attrs_mapping(cg)) {
    DynamicNodeAttrs result_attrs{
        /*task_type=*/std::nullopt,
        /*device_coord=*/std::nullopt,
        /*mapping=*/std::nullopt,
        /*op_attrs=*/pcg_op_attrs_from_compgraph_op_attrs(attrs.op_attrs),
        /*pcg_layer_guid=*/dynamic_layer_guid_t{layer},
        /*per_device_op_state=*/std::nullopt};

    std::unordered_map<DynamicTensorSlot, DynamicValueAttrs> result_inputs;
    for (auto const &[slot_name, tensor] : get_incoming_tensors(cg, layer)) {
      result_inputs.emplace(std::piecewise_construct,
                            std::forward_as_tuple(
                                /*slot_name=*/slot_name,
                                /*slot_tensor_role=*/std::nullopt),
                            std::forward_as_tuple(
                                /*tensor_guid=*/dynamic_tensor_guid_t{tensor},
                                /*parallel_tensor_shape=*/std::nullopt,
                                /*shard_coord=*/std::nullopt,
                                /*accessor=*/std::nullopt,
                                /*role=*/std::nullopt));
    }
    std::unordered_map<DynamicTensorSlot, DynamicValueAttrs> result_outputs;
    for (auto const &[slot_name, tensor] : get_outgoing_tensors(cg, layer)) {
      result_outputs.emplace(std::piecewise_construct,
                             std::forward_as_tuple(
                                 /*slot_name=*/slot_name,
                                 /*slot_tensor_role=*/std::nullopt),
                             std::forward_as_tuple(
                                 /*tensor_guid=*/dynamic_tensor_guid_t{tensor},
                                 /*parallel_tensor_shape=*/std::nullopt,
                                 /*shard_coord=*/std::nullopt,
                                 /*accessor=*/std::nullopt,
                                 /*role=*/std::nullopt));
    }

    result.invocations.emplace(result_inputs, result_attrs, result_outputs);
  }

  return result;
}

} // namespace FlexFlow
