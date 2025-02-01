#include "substitutions/output_graph/output_operator_attrs_assignment.h"
#include "substitutions/operator_pattern/get_attribute_map.h"
#include "substitutions/output_graph/materialize_operator_from_attrs_map.h"
#include "substitutions/output_graph/output_operator_attribute_expr.h"
#include "utils/containers/map_values.h"
#include "utils/containers/merge_maps.h"

namespace FlexFlow {

OutputOperatorAttrsAssignment output_operator_clone_node(PatternNode const &) {
  NOT_IMPLEMENTED();
}

PCGOperatorAttrs materialize_output_operator_from_attrs_assignment(
    OutputOperatorAttrsAssignment const &attrs_assignment,
    std::unordered_map<PatternNode, PCGOperatorAttrs> const &node_match) {

  std::unordered_map<OperatorAttributeKey, OperatorAttributeValue>
      template_attrs_map = [&]()
      -> std::unordered_map<OperatorAttributeKey, OperatorAttributeValue> {
    if (attrs_assignment.template_operator.has_value()) {
      PatternNode template_node = attrs_assignment.template_operator.value();
      PCGOperatorAttrs template_op_attrs = node_match.at(template_node);
      return get_attribute_map(template_op_attrs);
    } else {
      return {};
    }
  }();

  std::unordered_map<OperatorAttributeKey, OperatorAttributeValue>
      assignments_attrs_map = map_values(
          attrs_assignment.assignments,
          [&](OutputOperatorAttributeExpr const &expr) {
            return evaluate_output_operator_attribute_expr(expr, node_match);
          });

  std::unordered_map<OperatorAttributeKey, OperatorAttributeValue>
      joined_attrs_map =
          merge_map_right_dominates(template_attrs_map, assignments_attrs_map);

  return materialize_operator_from_attrs_map(joined_attrs_map);
}

std::pair<OperatorAttributeKey, OutputOperatorAttributeExpr>
    copy_attr_from_pattern_node(OperatorAttributeKey key,
                                PatternNode const &pattern_node) {
  return {key,
          OutputOperatorAttributeExpr{OutputOperatorAttrAccess{
              pattern_node, OperatorAttributeExpr{key}}}};
}

std::pair<OperatorAttributeKey, OutputOperatorAttributeExpr>
    set_attr_to_constant(OperatorAttributeKey key,
                         OperatorAttributeValue const &value) {
  return {
      key,
      OutputOperatorAttributeExpr{AttrConstant{value}},
  };
}

std::pair<OperatorAttributeKey, OutputOperatorAttributeExpr>
    set_op_type_attr(OperatorType op_type) {
  return set_attr_to_constant(OperatorAttributeKey::OP_TYPE,
                              OperatorAttributeValue{op_type});
}

} // namespace FlexFlow
