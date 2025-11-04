#include "task-spec/dynamic_graph/dynamic_open_dataflow_graph.h"

namespace FlexFlow {

bool full_dynamic_graph_satisfies(
  DynamicOpenDataflowGraph const &,
  std::function<bool(DynamicNodeAttrs const &)> const &,
  std::function<bool(DynamicValueAttrs const &)> const &) {

  NOT_IMPLEMENTED();
}

bool no_part_of_dynamic_graph_satisfies(
  DynamicOpenDataflowGraph const &,
  std::function<bool(DynamicNodeAttrs const &)> const &,
  std::function<bool(DynamicValueAttrs const &)> const &) {

  NOT_IMPLEMENTED();
}

} // namespace FlexFlow
