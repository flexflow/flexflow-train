#include "substitutions/substitution.h"
#include "substitutions/output_graph/output_graph_expr.h"
#include "substitutions/pcg_pattern.h"
#include "utils/bidict/algorithms/left_entries.h"
#include "utils/bidict/algorithms/right_entries.h"
#include "utils/containers/map_values.h"
#include "utils/graph/labelled_open_dataflow_graph/algorithms/find_isomorphism.h"
#include "utils/graph/labelled_open_dataflow_graph/algorithms/rewrite_node_labels.h"
#include "utils/graph/open_dataflow_graph/algorithms/open_dataflow_graph_isomorphism.dtg.h"
#include "utils/graph/open_dataflow_graph/algorithms/open_dataflow_graph_isomorphism.h"

namespace FlexFlow {

bool is_isomorphic_to(Substitution const &l, Substitution const &r) {
  OpenDataflowGraphIsomorphism pcg_pattern_isomorphism = ({
    std::optional<OpenDataflowGraphIsomorphism> maybe_isomorphism =
        find_isomorphism(l.pcg_pattern.raw_graph, r.pcg_pattern.raw_graph);

    if (!maybe_isomorphism.has_value()) {
      return false;
    }

    maybe_isomorphism.value();
  });

  auto l_from_r_pattern_node = [&](PatternNode const &r_node) {
    return PatternNode{
        pcg_pattern_isomorphism.node_mapping.at_r(r_node.raw_node),
    };
  };

  auto l_from_r_output_attrs_assignment =
      [&](OutputOperatorAttrsAssignment const &r_attrs) {
        std::optional<PatternNode> l_template_operator =
            transform(r_attrs.template_operator, l_from_r_pattern_node);
        std::unordered_map<OperatorAttributeKey, OutputOperatorAttributeExpr>
            l_assignments = map_values(
                r_attrs.assignments,
                [&](OutputOperatorAttributeExpr const &r_expr) {
                  return r_expr.visit<OutputOperatorAttributeExpr>(
                      overload{[&](AttrConstant const &) { return r_expr; },
                               [&](OutputOperatorAttrAccess const &r_acc) {
                                 return OutputOperatorAttributeExpr{
                                     OutputOperatorAttrAccess{
                                         l_from_r_pattern_node(r_acc.node),
                                         r_acc.attr_expr,
                                     },
                                 };
                               }});
                });
        return OutputOperatorAttrsAssignment{
            l_template_operator,
            l_assignments,
        };
      };

  OpenDataflowGraphIsomorphism output_graph_expr_isomorphism = ({
    std::optional<OpenDataflowGraphIsomorphism> maybe_isomorphism =
        find_isomorphism(
            l.output_graph_expr.raw_graph,
            rewrite_node_labels(
                r.output_graph_expr.raw_graph,
                [&](Node const &, OutputOperatorAttrsAssignment const &a) {
                  return l_from_r_output_attrs_assignment(a);
                }));
    if (!maybe_isomorphism.has_value()) {
      return false;
    }

    maybe_isomorphism.value();
  });

  auto l_from_r_pattern_input = [&](PatternInput const &r_input) {
    return PatternInput{
        pcg_pattern_isomorphism.input_mapping.at_r(
            r_input.raw_dataflow_graph_input),
    };
  };

  auto l_from_r_output_graph_input = [&](OutputGraphExprInput const &r_input) {
    return OutputGraphExprInput{
        output_graph_expr_isomorphism.input_mapping.at_r(
            r_input.raw_dataflow_graph_input),
    };
  };

  auto l_from_r_pattern_output = [&](PatternNodeOutput const &r_output) {
    return PatternNodeOutput{
        isomorphism_map_l_dataflow_output_from_r(pcg_pattern_isomorphism,
                                                 r_output.raw_dataflow_output),
    };
  };

  auto l_from_r_output_graph_output =
      [&](OutputGraphExprNodeOutput const &r_output) {
        return OutputGraphExprNodeOutput{
            isomorphism_map_l_dataflow_output_from_r(
                output_graph_expr_isomorphism, r_output.raw_dataflow_output),
        };
      };

  bidict<PatternInput, OutputGraphExprInput> l_input_mapping_from_r =
      transform(r.inputs_mapping,
                [&](PatternInput const &r_p, OutputGraphExprInput const &r_o) {
                  return std::pair{
                      l_from_r_pattern_input(r_p),
                      l_from_r_output_graph_input(r_o),
                  };
                });
  if (l_input_mapping_from_r != l.inputs_mapping) {
    return false;
  }

  bidict<PatternNodeOutput, OutputGraphExprNodeOutput> l_output_mapping_from_r =
      transform(r.outputs_mapping,
                [&](PatternNodeOutput const &r_p,
                    OutputGraphExprNodeOutput const &r_o) {
                  return std::pair{
                      l_from_r_pattern_output(r_p),
                      l_from_r_output_graph_output(r_o),
                  };
                });
  if (l_output_mapping_from_r != l.outputs_mapping) {
    return false;
  }

  return true;
}

bool is_valid_substitution(Substitution const &sub) {
  {
    std::unordered_set<PatternInput> pattern_inputs =
        get_inputs(sub.pcg_pattern);
    std::unordered_set<PatternInput> mapped_inputs =
        left_entries(sub.inputs_mapping);

    if (pattern_inputs != mapped_inputs) {
      return false;
    }
  }

  {
    std::unordered_set<OutputGraphExprInput> output_graph_inputs =
        get_inputs(sub.output_graph_expr);
    std::unordered_set<OutputGraphExprInput> mapped_inputs =
        right_entries(sub.inputs_mapping);

    if (output_graph_inputs != mapped_inputs) {
      return false;
    }
  }

  if (get_nodes(sub.pcg_pattern).empty()) {
    return false;
  }

  if (get_nodes(sub.output_graph_expr).empty()) {
    return false;
  }

  return true;
}

} // namespace FlexFlow
