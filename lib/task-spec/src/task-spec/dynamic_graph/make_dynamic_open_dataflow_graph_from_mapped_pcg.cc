#include "task-spec/dynamic_graph/make_dynamic_open_dataflow_graph_from_mapped_pcg.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "op-attrs/pcg_operator_attrs.h"
#include "pcg/mapped_parallel_computation_graph/mapped_operator_task_group.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph.h"
#include "pcg/parallel_computation_graph/parallel_tensor_attrs.dtg.h"
#include "task-spec/dynamic_graph/dynamic_layer_guid_t.dtg.h"
#include "task-spec/dynamic_graph/dynamic_open_dataflow_graph.h"
#include "task-spec/dynamic_graph/dynamic_tensor_role.h"
#include "utils/containers/generate_map.h"
#include <optional>
#include <unordered_map>
#include <utility>

namespace FlexFlow {

DynamicOpenDataflowGraph make_dynamic_open_dataflow_graph_from_mapped_pcg(
    MappedParallelComputationGraph const &mpcg) {
  DynamicOpenDataflowGraph result = make_empty_dynamic_open_dataflow_graph();

  for (auto const &[layer, attrs] :
       get_parallel_layer_attrs_mapping(mpcg.pcg)) {
    MappedOperatorTaskGroup node_mapping = mpcg.mapped_tasks.at(layer);
    DynamicNodeAttrs result_attrs{
        /*task_type=*/std::nullopt,
        /*device_coord=*/std::nullopt,
        /*mapping=*/node_mapping,
        /*op_attrs=*/TrainingOperationAttrs{attrs.op_attrs},
        /*pcg_layer_guid=*/dynamic_layer_guid_t{layer},
        /*per_device_op_state=*/std::nullopt,
    };

    std::unordered_map<DynamicTensorSlot, DynamicValueAttrs> result_inputs =
        transform(get_incoming_tensors(mpcg.pcg, layer),
                  [&](TensorSlotName const &slot_name,
                      parallel_tensor_guid_t const &tensor) {
                    ParallelTensorAttrs attrs =
                        get_parallel_tensor_attrs(mpcg.pcg, tensor);
                    return std::pair<DynamicTensorSlot, DynamicValueAttrs>{
                        DynamicTensorSlot{
                            /*slot_name=*/slot_name,
                            /*slot_tensor_role=*/std::nullopt,
                        },
                        DynamicValueAttrs{
                            /*tensor_guid=*/dynamic_tensor_guid_t{tensor},
                            /*parallel_tensor_shape=*/attrs.shape,
                            /*shard_coord=*/std::nullopt,
                            /*mapping=*/
                            get_tensor_bindings_for_slot_name(node_mapping,
                                                              slot_name),
                            /*accessor=*/std::nullopt,
                            /*role=*/std::nullopt,
                        },
                    };
                  });
    std::unordered_map<DynamicTensorSlot, DynamicValueAttrs> result_outputs =
        transform(get_outgoing_tensors(mpcg.pcg, layer),
                  [&](TensorSlotName const &slot_name,
                      parallel_tensor_guid_t const &tensor) {
                    ParallelTensorAttrs attrs =
                        get_parallel_tensor_attrs(mpcg.pcg, tensor);
                    return std::pair<DynamicTensorSlot, DynamicValueAttrs>{
                        DynamicTensorSlot{
                            /*slot_name=*/slot_name,
                            /*slot_tensor_role=*/std::nullopt,
                        },
                        DynamicValueAttrs{
                            /*tensor_guid=*/dynamic_tensor_guid_t{tensor},
                            /*parallel_tensor_shape=*/attrs.shape,
                            /*shard_coord=*/std::nullopt,
                            /*mapping=*/
                            get_tensor_bindings_for_slot_name(node_mapping,
                                                              slot_name),
                            /*accessor=*/std::nullopt,
                            /*role=*/std::nullopt,
                        },
                    };
                  });

    result.invocations.emplace(result_inputs, result_attrs, result_outputs);
  }

  return result;
}

} // namespace FlexFlow
