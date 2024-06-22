// THIS FILE WAS AUTO-GENERATED BY proj. DO NOT MODIFY IT!
// If you would like to modify this datatype, instead modify
// lib/substitutions/include/substitutions/unlabelled/output_pattern_edge.struct.toml
/* proj-data
{
  "generated_from": "3222696e351c3e203e008714245c737f"
}
*/

#include "substitutions/unlabelled/output_pattern_edge.dtg.h"

#include "utils/graph.h"

namespace FlexFlow {
OutputPatternEdge::OutputPatternEdge(
    ::FlexFlow::OutputMultiDiEdge const &raw_edge)
    : raw_edge(raw_edge) {}
bool OutputPatternEdge::operator==(OutputPatternEdge const &other) const {
  return std::tie(this->raw_edge) == std::tie(other.raw_edge);
}
bool OutputPatternEdge::operator!=(OutputPatternEdge const &other) const {
  return std::tie(this->raw_edge) != std::tie(other.raw_edge);
}
bool OutputPatternEdge::operator<(OutputPatternEdge const &other) const {
  return std::tie(this->raw_edge) < std::tie(other.raw_edge);
}
bool OutputPatternEdge::operator>(OutputPatternEdge const &other) const {
  return std::tie(this->raw_edge) > std::tie(other.raw_edge);
}
bool OutputPatternEdge::operator<=(OutputPatternEdge const &other) const {
  return std::tie(this->raw_edge) <= std::tie(other.raw_edge);
}
bool OutputPatternEdge::operator>=(OutputPatternEdge const &other) const {
  return std::tie(this->raw_edge) >= std::tie(other.raw_edge);
}
} // namespace FlexFlow

namespace std {
size_t hash<FlexFlow::OutputPatternEdge>::operator()(
    ::FlexFlow::OutputPatternEdge const &x) const {
  size_t result = 0;
  result ^= std::hash<::FlexFlow::OutputMultiDiEdge>{}(x.raw_edge) +
            0x9e3779b9 + (result << 6) + (result >> 2);
  return result;
}
} // namespace std