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

  std::vector<PatternValue> add_pattern_node(
      OperatorAttributePattern const &node_pattern,
      std::vector<PatternValue> const &inputs,
      std::vector<TensorAttributePattern> const &output_patterns,
      std::optional<std::string> const &name = std::nullopt);

  std::vector<OutputGraphExprValue>
      add_output_graph_node(OutputOperatorAttrsAssignment const &node_expr,
                            std::vector<OutputGraphExprValue> const &inputs,
                            nonnegative_int num_outputs);

  PatternNode pattern_node_named(std::string const &) const;
  PatternInput pattern_input_named(std::string const &) const;

  Substitution get_substitution() const;

private:
  LabelledOpenDataflowGraph<OperatorAttributePattern, TensorAttributePattern>
      pattern_g;
  LabelledOpenDataflowGraph<OutputOperatorAttrsAssignment, std::monostate>
      output_g;
  bidict<PatternInput, OutputGraphExprInput> input_mapping;
  bidict<PatternNode, std::string> pattern_node_names;
  bidict<PatternInput, std::string> pattern_input_names;
  bidict<PatternNodeOutput, OutputGraphExprNodeOutput> output_mapping;
};

} // namespace FlexFlow

#endif
