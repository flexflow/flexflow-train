#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_RENDER_DOT_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_RENDER_DOT_H

#include "utils/graph/labelled_open_dataflow_graph/labelled_open_dataflow_graph_view.h"
#include <string>
#include <unordered_map>

namespace FlexFlow {

std::string escape_dot_string(std::string const &);
std::string render_dot_node_attrs(
    std::unordered_map<std::string, std::string> const &attrs);
std::string render_dot(
    LabelledDataflowGraphView<std::unordered_map<std::string, std::string>,
                              std::string> const &);

} // namespace FlexFlow

#endif
