// THIS FILE WAS AUTO-GENERATED BY proj. DO NOT MODIFY IT!
// If you would like to modify this datatype, instead modify
// lib/substitutions/include/substitutions/output_graph/output_operator_attrs_assignment.struct.toml
/* proj-data
{
  "generated_from": "bbfb309c5a39a729da23dace4df4a9de"
}
*/

#ifndef _FLEXFLOW_LIB_SUBSTITUTIONS_INCLUDE_SUBSTITUTIONS_OUTPUT_GRAPH_OUTPUT_OPERATOR_ATTRS_ASSIGNMENT_DTG_H
#define _FLEXFLOW_LIB_SUBSTITUTIONS_INCLUDE_SUBSTITUTIONS_OUTPUT_GRAPH_OUTPUT_OPERATOR_ATTRS_ASSIGNMENT_DTG_H

#include "fmt/format.h"
#include "substitutions/operator_pattern/operator_attribute_key.dtg.h"
#include "substitutions/output_graph/output_operator_attribute_expr.dtg.h"
#include <functional>
#include <ostream>
#include <tuple>
#include <unordered_map>

namespace FlexFlow {
struct OutputOperatorAttrsAssignment {
  OutputOperatorAttrsAssignment() = delete;
  explicit OutputOperatorAttrsAssignment(
      std::unordered_map<::FlexFlow::OperatorAttributeKey,
                         ::FlexFlow::OutputOperatorAttributeExpr> const
          &assignments);

  bool operator==(OutputOperatorAttrsAssignment const &) const;
  bool operator!=(OutputOperatorAttrsAssignment const &) const;
  std::unordered_map<::FlexFlow::OperatorAttributeKey,
                     ::FlexFlow::OutputOperatorAttributeExpr>
      assignments;
};
} // namespace FlexFlow

namespace std {
template <>
struct hash<::FlexFlow::OutputOperatorAttrsAssignment> {
  size_t operator()(::FlexFlow::OutputOperatorAttrsAssignment const &) const;
};
} // namespace std

namespace FlexFlow {
std::string format_as(OutputOperatorAttrsAssignment const &);
std::ostream &operator<<(std::ostream &, OutputOperatorAttrsAssignment const &);
} // namespace FlexFlow

#endif // _FLEXFLOW_LIB_SUBSTITUTIONS_INCLUDE_SUBSTITUTIONS_OUTPUT_GRAPH_OUTPUT_OPERATOR_ATTRS_ASSIGNMENT_DTG_H