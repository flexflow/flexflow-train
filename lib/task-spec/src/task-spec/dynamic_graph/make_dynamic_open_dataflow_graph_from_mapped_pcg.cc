#include "task-spec/dynamic_graph/make_dynamic_open_dataflow_graph_from_mapped_pcg.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "op-attrs/pcg_operator_attrs.h"
#include "pcg/mapped_parallel_computation_graph/mapped_parallel_computation_graph.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph_edge.h"
#include "pcg/parallel_computation_graph/parallel_tensor_attrs.dtg.h"
#include "pcg/parallel_computation_graph/parallel_tensor_use_t.h"
#include "task-spec/dynamic_graph/dynamic_layer_guid_t.dtg.h"
#include "task-spec/dynamic_graph/dynamic_open_dataflow_graph.h"
#include "task-spec/dynamic_graph/dynamic_tensor_role.h"
#include "utils/bidict/algorithms/merge_disjoint_bidicts.h"
#include "utils/containers/generate_map.h"
#include "utils/containers/get_only.h"
#include "utils/containers/map_keys_and_values.h"
#include "utils/containers/require_only_key.h"
#include "utils/containers/transform_pairs.h"
#include <optional>
#include <unordered_map>
#include <utility>

namespace FlexFlow {

static bidict<ParallelTensorSpaceCoordinate, MachineSpaceCoordinate>
    get_input_mapping_for_replicate(
        MappedParallelComputationGraph const &mpcg,
        parallel_layer_guid_t const &replicate_layer) {

  ASSERT(mpcg_get_pcg_op_attrs(mpcg, replicate_layer).is_parallel_replicate());

  auto [input_slot_name, input_edge] =
      get_only(mpcg_get_incoming_edges(mpcg, replicate_layer));

  parallel_layer_guid_t producer_layer = get_src_layer(input_edge);
  TensorSlotName producer_slot = get_src_layer_output_slot_name(input_edge);

  return get_tensor_bindings_for_slot_name(
      /*task_group=*/mpcg_get_mapping_for_layer(mpcg, producer_layer),
      /*slot_name=*/producer_slot);
}

static bidict<ParallelTensorSpaceCoordinate, MachineSpaceCoordinate>
    build_replicated_output_mapping(
        MappedParallelComputationGraph const &mpcg,
        parallel_tensor_guid_t const &output_tensor_guid) {

  std::unordered_set<parallel_tensor_use_t> consumers =
      mpcg_get_parallel_tensor_uses(mpcg, output_tensor_guid);
  ASSERT(!consumers.empty());

  // union all consumer bindings — each consumer shard maps to a distinct
  // (discard_copy, machine) pair since replicas are always on different machines
  bidict<ParallelTensorSpaceCoordinate, MachineSpaceCoordinate> result =
      merge_disjoint_bidicts(transform(
          consumers,
          [&](parallel_tensor_use_t const &use)
              -> bidict<ParallelTensorSpaceCoordinate, MachineSpaceCoordinate> {
            parallel_layer_guid_t consumer_layer =
                parallel_tensor_use_get_layer(use);
            TensorSlotName slot_name = parallel_tensor_use_get_slot(use);

            MappedOperatorTaskGroup consumer_mapping =
                mpcg_get_mapping_for_layer(mpcg, consumer_layer);
            bidict<ParallelTensorSpaceCoordinate, MachineSpaceCoordinate>
                binding = get_tensor_bindings_for_slot_name(consumer_mapping,
                                                            slot_name);

            return binding;
          }));

  return result;
}

static DynamicNodeInvocation
    build_replicate_invocation(parallel_layer_guid_t const &layer,
                               ReplicateAttrs const &attrs,
                               MappedParallelComputationGraph const &mpcg) {

  ManyToOne<TensorSlotName, parallel_tensor_guid_t> incoming =
      mpcg_get_incoming_tensors(mpcg, layer);
  TensorSlotName input_slot_name = TensorSlotName::INPUT;
  parallel_tensor_guid_t input_tensor_guid =
      require_only_key(incoming.l_to_r(), input_slot_name);
  ParallelTensorAttrs input_attrs =
      mpcg_get_parallel_tensor_attrs(mpcg, input_tensor_guid);

  bidict<TensorSlotName, parallel_tensor_guid_t> outgoing =
      mpcg_get_outgoing_tensors(mpcg, layer);
  TensorSlotName output_slot_name = TensorSlotName::OUTPUT;
  parallel_tensor_guid_t output_tensor_guid =
      require_only_key(outgoing.l_to_r(), output_slot_name);
  ParallelTensorAttrs output_attrs =
      mpcg_get_parallel_tensor_attrs(mpcg, output_tensor_guid);

  bidict<ParallelTensorSpaceCoordinate, MachineSpaceCoordinate> input_mapping =
      get_input_mapping_for_replicate(mpcg, layer);

  DynamicValueAttrs input_value{
      /*tensor_guid=*/dynamic_tensor_guid_t{input_tensor_guid},
      /*parallel_tensor_shape=*/input_attrs.shape,
      /*shard_coord=*/std::nullopt,
      /*mapping=*/input_mapping,
      /*accessor=*/std::nullopt,
      /*role=*/std::nullopt,
  };

  DynamicValueAttrs output_value{
      /*tensor_guid=*/dynamic_tensor_guid_t{output_tensor_guid},
      /*parallel_tensor_shape=*/output_attrs.shape,
      /*shard_coord=*/std::nullopt,
      /*mapping=*/build_replicated_output_mapping(mpcg, output_tensor_guid),
      /*accessor=*/std::nullopt,
      /*role=*/std::nullopt,
  };

  DynamicNodeAttrs node_attrs{
      /*task_type=*/std::nullopt,
      /*device_coord=*/std::nullopt,
      /*mapping=*/std::nullopt,
      /*op_attrs=*/TrainingOperationAttrs{PCGOperatorAttrs{attrs}},
      /*pcg_layer_guid=*/dynamic_layer_guid_t{layer},
      /*per_device_op_state=*/std::nullopt,
  };

  DynamicNodeInvocation invocation_node{
      /*inputs=*/{
          {
              DynamicTensorSlot{input_slot_name, std::nullopt},
              input_value,
          },
      },
      /*node_attrs=*/node_attrs,
      /*outputs=*/
      {
          {
              DynamicTensorSlot{output_slot_name, std::nullopt},
              output_value,
          },
      },
  };

  return invocation_node;
}

DynamicOpenDataflowGraph make_dynamic_open_dataflow_graph_from_mapped_pcg(
    MappedParallelComputationGraph const &mpcg) {

  ParallelComputationGraph pcg = pcg_from_mpcg(mpcg);

  auto mk_invocation =
      [&](parallel_layer_guid_t layer,
          ParallelLayerAttrs const &attrs) -> DynamicNodeInvocation {
    if (attrs.op_attrs.is_parallel_replicate()) {
      // build replicate invocation
      DynamicNodeInvocation repl_inv = build_replicate_invocation(
          layer, attrs.op_attrs.require_parallel_replicate(), mpcg);
      return repl_inv;
    } else {
      DynamicNodeAttrs result_attrs{
          /*task_type=*/std::nullopt,
          /*device_coord=*/std::nullopt,
          /*mapping=*/mpcg_get_mapping_for_layer(mpcg, layer),
          /*op_attrs=*/TrainingOperationAttrs{attrs.op_attrs},
          /*pcg_layer_guid=*/dynamic_layer_guid_t{layer},
          /*per_device_op_state=*/std::nullopt,
      };

      auto mk_slot = [](TensorSlotName const &slot_name) -> DynamicTensorSlot {
        return DynamicTensorSlot{
            /*slot_name=*/slot_name,
            /*slot_tensor_role=*/std::nullopt,
        };
      };

      auto mk_value_attrs =
          [&](parallel_tensor_guid_t const &tensor) -> DynamicValueAttrs {
        ParallelTensorAttrs attrs = get_parallel_tensor_attrs(pcg, tensor);

        return DynamicValueAttrs{
            /*tensor_guid=*/dynamic_tensor_guid_t{tensor},
            /*parallel_tensor_shape=*/attrs.shape,
            /*shard_coord=*/std::nullopt,
            /*mapping=*/std::nullopt,
            /*accessor=*/std::nullopt,
            /*role=*/std::nullopt,
        };
      };

      std::unordered_map<DynamicTensorSlot, DynamicValueAttrs> result_inputs =
          map_keys_and_values(
              get_incoming_tensors(pcg, layer), mk_slot, mk_value_attrs);

      std::unordered_map<DynamicTensorSlot, DynamicValueAttrs> result_outputs =
          map_keys_and_values(
              get_outgoing_tensors(pcg, layer), mk_slot, mk_value_attrs);

      DynamicNodeInvocation invocation = DynamicNodeInvocation{
          /*inputs=*/result_inputs,
          /*node_attrs=*/result_attrs,
          /*outputs=*/result_outputs,
      };

      return invocation;
    };
  };

  return dynamic_open_dataflow_graph_from_invocation_set(transform_pairs(
      unordered_set_of(get_parallel_layer_attrs_mapping(pcg)), mk_invocation));
}

} // namespace FlexFlow
