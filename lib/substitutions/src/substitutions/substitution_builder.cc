#include "substitutions/substitution_builder.h"
#include "substitutions/output_graph/output_graph_expr_value.h"
#include "substitutions/substitution.h"
#include "substitutions/unlabelled/pattern_value.h"
#include "utils/containers/repeat_element.h"
#include "utils/graph/instances/unordered_set_labelled_open_dataflow_graph.h"
#include "utils/overload.h"

namespace FlexFlow {

SubstitutionBuilder::SubstitutionBuilder()
    : pattern_g(LabelledOpenDataflowGraph<OperatorAttributePattern,
                                          TensorAttributePattern>::
                    create<UnorderedSetLabelledOpenDataflowGraph<
                        OperatorAttributePattern,
                        TensorAttributePattern>>()),
      output_g(LabelledOpenDataflowGraph<OutputOperatorAttrsAssignment,
                                         std::monostate>::
                   create<UnorderedSetLabelledOpenDataflowGraph<
                       OutputOperatorAttrsAssignment,
                       std::monostate>>()) {}

std::pair<PatternValue, OutputGraphExprValue> SubstitutionBuilder::add_input(
    TensorAttributePattern const &input_tensor_pattern,
    std::optional<std::string> const &name) {
  PatternInput pattern_input = PatternInput{
      this->pattern_g.add_input(input_tensor_pattern),
  };

  OutputGraphExprInput output_graph_expr_input = OutputGraphExprInput{
      this->output_g.add_input(std::monostate{}),
  };

  this->input_mapping.equate(pattern_input, output_graph_expr_input);

  if (name.has_value()) {
    this->pattern_input_names.equate(pattern_input, name.value());
  }

  return {
      PatternValue{pattern_input},
      OutputGraphExprValue{output_graph_expr_input},
  };
}

std::vector<PatternValue> SubstitutionBuilder::add_pattern_node(
    OperatorAttributePattern const &node_pattern,
    std::vector<PatternValue> const &inputs,
    std::vector<TensorAttributePattern> const &output_patterns,
    std::optional<std::string> const &maybe_name) {
  NodeAddedResult node_added = this->pattern_g.add_node(
      node_pattern,
      transform(inputs, raw_open_dataflow_value_from_pattern_value),
      output_patterns);

  if (maybe_name.has_value()) {
    std::string name = maybe_name.value();

    if (this->pattern_node_names.contains_r(name)) {
      throw mk_runtime_error(fmt::format("Attempted to name node {}, but a "
                                         "node with that name already exists!",
                                         name));
    }

    this->pattern_node_names.equate(PatternNode{node_added.node}, name);
  }

  return transform(node_added.outputs, [](DataflowOutput const &o) {
    return pattern_value_from_raw_open_dataflow_value(OpenDataflowValue{o});
  });
}

std::vector<OutputGraphExprValue> SubstitutionBuilder::add_output_graph_node(
    OutputOperatorAttrsAssignment const &node_expr,
    std::vector<OutputGraphExprValue> const &inputs,
    nonnegative_int num_outputs) {
  NodeAddedResult node_added = this->output_g.add_node(
      node_expr,
      transform(inputs, raw_open_dataflow_value_from_output_graph_expr_value),
      repeat_element(/*num_times=*/num_outputs, /*element=*/std::monostate{}));

  return transform(node_added.outputs, [](DataflowOutput const &o) {
    return output_graph_expr_value_from_raw_open_dataflow_value(
        OpenDataflowValue{o});
  });
}

void SubstitutionBuilder::equate_outputs(
    PatternValue const &maybe_pattern_output,
    OutputGraphExprValue const &maybe_output_graph_expr_output) {
  PatternNodeOutput pattern_output =
      maybe_pattern_output.visit<PatternNodeOutput>(overload{
          [](PatternNodeOutput const &o) { return o; },
          [&](PatternInput const &) -> PatternNodeOutput {
            throw mk_runtime_error(fmt::format(
                "SubstitutionBuilder::equate_outputs expected a PatternValue "
                "holding a PatternNodeOutput, but received {}",
                maybe_pattern_output));
          },
      });

  OutputGraphExprNodeOutput output_graph_expr_output =
      maybe_output_graph_expr_output.visit<OutputGraphExprNodeOutput>(overload{
          [](OutputGraphExprNodeOutput const &o) { return o; },
          [&](OutputGraphExprInput const &) -> OutputGraphExprNodeOutput {
            throw mk_runtime_error(
                fmt::format("SubstitutionBuilder::equate_outputs expected an "
                            "OutputGraphExprValue holding a "
                            "OutputGraphExprNodeOutput, but received {}",
                            maybe_output_graph_expr_output));
          },
      });

  if (this->output_mapping.contains_l(pattern_output)) {
    throw mk_runtime_error(
        fmt::format("SubstitutionBuilder::equate_outputs expected a "
                    "PatternValue holding a PatternValueOutput"
                    "that is not contained in the output_mapping forward graph,"
                    "but received {}",
                    pattern_output));
  }
  if (this->output_mapping.contains_r(output_graph_expr_output)) {
    throw mk_runtime_error(fmt::format(
        "SubstitutionBuilder::output_graph_expr_output expected a "
        "OutputGraphExprValue holding a OutputGraphExprNodeOutput"
        "that is not contained in the output_mapping backward graph,"
        "but received {}",
        output_graph_expr_output));
  }

  this->output_mapping.equate(pattern_output, output_graph_expr_output);
}

PatternNode
    SubstitutionBuilder::pattern_node_named(std::string const &name) const {
  return this->pattern_node_names.at_r(name);
}

PatternInput
    SubstitutionBuilder::pattern_input_named(std::string const &name) const {
  return this->pattern_input_names.at_r(name);
}

Substitution SubstitutionBuilder::get_substitution() const {
  Substitution result = Substitution{
      PCGPattern{this->pattern_g},
      OutputGraphExpr{this->output_g},
      this->input_mapping,
      this->output_mapping,
  };

  if (!is_valid_substitution(result)) {
    throw mk_runtime_error(
        "get_substitution cannot return a Substitution, as the Substitution is "
        "currently invalid. Ensure you have finished constructing the "
        "Substitution and have mapped all of the outputs.");
  }

  return result;
}

} // namespace FlexFlow
