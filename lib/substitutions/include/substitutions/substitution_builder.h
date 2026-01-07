#ifndef _FLEXFLOW_LIB_SUBSTITUTIONS_INCLUDE_SUBSTITUTIONS_SUBSTITUTION_BUILDER_H
#define _FLEXFLOW_LIB_SUBSTITUTIONS_INCLUDE_SUBSTITUTIONS_SUBSTITUTION_BUILDER_H

#include "substitutions/output_graph/output_graph_expr_value.dtg.h"
#include "substitutions/substitution.dtg.h"
#include "substitutions/unlabelled/pattern_value.dtg.h"
#include <tl/expected.hpp>

namespace FlexFlow {

struct SubstitutionBuilder {
public:
  SubstitutionBuilder();

  std::pair<PatternValue, OutputGraphExprValue>
      add_input(TensorAttributePattern const &,
                std::optional<std::string> const &name = std::nullopt);
  void equate_outputs(PatternValue const &, OutputGraphExprValue const &);

  std::unordered_map<TensorSlotName, PatternValue> add_pattern_node(
      OperatorAttributePattern const &node_pattern,
      std::unordered_map<TensorSlotName, PatternValue> const &inputs,
      std::unordered_map<TensorSlotName, TensorAttributePattern> const
          &output_patterns,
      std::optional<std::string> const &name = std::nullopt);

  std::unordered_map<TensorSlotName, OutputGraphExprValue>
      add_output_graph_node(
          OutputOperatorAttrsAssignment const &node_expr,
          std::unordered_map<TensorSlotName, OutputGraphExprValue> const
              &inputs,
          std::unordered_set<TensorSlotName> const &output_slots);

  PatternNode pattern_node_named(std::string const &) const;
  PatternInput pattern_input_named(std::string const &) const;

  Substitution get_substitution() const;

private:
  int get_fresh_graph_input_name();

private:
  LabelledOpenKwargDataflowGraph<OperatorAttributePattern,
                                 TensorAttributePattern,
                                 int,
                                 TensorSlotName>
      pattern_g;
  LabelledOpenKwargDataflowGraph<OutputOperatorAttrsAssignment,
                                 std::monostate,
                                 int,
                                 TensorSlotName>
      output_g;
  bidict<PatternInput, OutputGraphExprInput> input_mapping;
  bidict<PatternNode, std::string> pattern_node_names;
  bidict<PatternInput, std::string> pattern_input_names;
  bidict<PatternNodeOutput, OutputGraphExprNodeOutput> output_mapping;
  int next_graph_input_id;
};

} // namespace FlexFlow

#endif
