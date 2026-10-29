#include "op-attrs/get_operator_space_to_parallel_tensor_space_mappings.h"
#include "op-attrs/get_incoming_tensor_roles.h"
#include "op-attrs/get_standard_operator_task_group.h"
#include "op-attrs/get_tensor_roles.h"
#include "op-attrs/incoming_tensor_role.h"
#include "op-attrs/ops/element_binary.h"
#include "op-attrs/ops/element_unary.h"
#include "op-attrs/ops/input.h"
#include "op-attrs/ops/linear.h"
#include "op-attrs/ops/transpose.h"
#include "op-attrs/ops/weight.h"
#include "op-attrs/pcg_operator_attrs.h"
#include "utils/containers/filter_values.h"
#include "utils/containers/filtrans.h"
#include "utils/containers/get_only.h"
#include "utils/containers/keys.h"
#include "utils/containers/merge_disjoint_maps.h"
#include "utils/containers/require_only_key.h"
#include "utils/containers/require_two_keys.h"
#include "utils/containers/zip_values_strict.h"
#include "utils/overload.h"

namespace FlexFlow {

std::map<TensorSlotName, OperatorSpaceToParallelTensorSpaceMapping>
    get_operator_to_incoming_mappings(
        ComputationGraphOpAttrs const &comp_graph_op_attrs,
        std::map<TensorSlotName, ParallelTensorDimDegrees> const
            &inputs_degrees) {

  return binary_merge_disjoint_maps(
      get_operator_to_input_mappings(comp_graph_op_attrs, inputs_degrees),
      get_operator_to_weight_mappings(comp_graph_op_attrs, inputs_degrees));
}

std::map<TensorSlotName, OperatorSpaceToParallelTensorSpaceMapping>
    get_operator_to_incoming_mappings_for_role(
        ComputationGraphOpAttrs const &attrs,
        std::map<TensorSlotName, ParallelTensorDimDegrees> const
            &inputs_degrees,
        IncomingTensorRole incoming_tensor_role) {
  return get_operator_to_ptensor_mappings_for_role(
      attrs,
      inputs_degrees,
      tensor_role_from_incoming_tensor_role(incoming_tensor_role));
}

std::map<TensorSlotName, OperatorSpaceToParallelTensorSpaceMapping>
    get_operator_to_input_mappings(
        ComputationGraphOpAttrs const &attrs,
        std::map<TensorSlotName, ParallelTensorDimDegrees> const
            &inputs_degrees) {
  return get_operator_to_ptensor_mappings_for_role(
      attrs, inputs_degrees, TensorRole::INPUT);
}

std::map<TensorSlotName, OperatorSpaceToParallelTensorSpaceMapping>
    get_operator_to_weight_mappings(
        ComputationGraphOpAttrs const &attrs,
        std::map<TensorSlotName, ParallelTensorDimDegrees> const
            &inputs_degrees) {

  return get_operator_to_ptensor_mappings_for_role(
      attrs, inputs_degrees, TensorRole::WEIGHT);
}

std::map<TensorSlotName, OperatorSpaceToParallelTensorSpaceMapping>
    get_operator_to_output_mappings(
        ComputationGraphOpAttrs const &comp_graph_op_attrs,
        std::map<TensorSlotName, ParallelTensorDimDegrees> const
            &inputs_degrees) {
  return get_operator_to_ptensor_mappings_for_role(
      comp_graph_op_attrs, inputs_degrees, TensorRole::OUTPUT);
}

std::map<TensorSlotName, OperatorSpaceToParallelTensorSpaceMapping>
    get_operator_to_ptensor_mappings_for_role(
        ComputationGraphOpAttrs const &attrs,
        std::map<TensorSlotName, ParallelTensorDimDegrees> const
            &inputs_degrees,
        TensorRole role) {

  StandardOperatorTaskGroup task_group =
      get_standard_operator_task_group(attrs, inputs_degrees);

  PCGOperatorAttrs pcg_op_attrs = pcg_op_attrs_from_compgraph_op_attrs(attrs);

  std::set<TensorSlotName> slot_names_for_role =
      keys(filter_values(get_tensor_roles(pcg_op_attrs),
                         [&](TensorRole r) -> bool { return r == role; }));

  return generate_map(
      slot_names_for_role,
      [&](TensorSlotName slot_name)
          -> OperatorSpaceToParallelTensorSpaceMapping {
        return standard_operator_task_group_get_operator_to_ptensor_mapping(
            task_group, slot_name);
      });
}

std::map<TensorSlotName, OperatorSpaceToParallelTensorSpaceMapping>
    get_operator_to_ptensor_mappings(
        ComputationGraphOpAttrs const &attrs,
        std::map<TensorSlotName, ParallelTensorDimDegrees> const
            &inputs_degrees) {
  return merge_disjoint_maps(std::vector{
      get_operator_to_input_mappings(attrs, inputs_degrees),
      get_operator_to_weight_mappings(attrs, inputs_degrees),
      get_operator_to_output_mappings(attrs, inputs_degrees),
  });
}

} // namespace FlexFlow
