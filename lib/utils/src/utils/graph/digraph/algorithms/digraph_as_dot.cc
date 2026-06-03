#include "utils/graph/digraph/algorithms/digraph_as_dot.h"
#include "utils/dot/dot_file.h"
#include "utils/dot/dot_html_from_json.h"
#include "utils/graph/digraph/algorithms/get_edges.h"
#include "utils/graph/node/algorithms.h"

namespace FlexFlow {

std::string digraph_as_dot(
    DiGraphView const &g,
    std::function<nlohmann::json(Node const &)> const &get_node_label) {
  std::ostringstream oss;
  DotFile<std::string> dot = DotFile<std::string>{oss};

  auto get_node_name = [](Node const &n) {
    return fmt::format("n{}", n.raw_uid);
  };

  for (Node const &n : get_nodes(g)) {
    dot.add_html_node(get_node_name(n),
                      dot_html_table_from_json(get_node_label(n)));
  }

  for (DirectedEdge const &e : get_edges(g)) {
    dot.add_edge(get_node_name(e.src), get_node_name(e.dst));
  }

  dot.close();
  return oss.str();
}

} // namespace FlexFlow
