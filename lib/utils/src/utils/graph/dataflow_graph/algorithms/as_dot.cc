#include "utils/graph/dataflow_graph/algorithms/as_dot.h"
#include "utils/containers/generate_map.h"
#include "utils/containers/map_keys.h"
#include "utils/dot_file.h"
#include "utils/graph/dataflow_graph/algorithms.h"
#include "utils/graph/dataflow_graph/algorithms/view_as_open_dataflow_graph.h"
#include "utils/graph/labelled_open_dataflow_graph/algorithms/with_labelling.h"
#include "utils/graph/node/algorithms.h"
#include "utils/graph/render_dot.h"
#include "utils/record_formatter.h"

namespace FlexFlow {

std::string as_dot(DataflowGraphView const &g) {
  auto get_node_attrs = [](Node const &) {
    return std::unordered_map<std::string, std::string>{};
  };

  std::unordered_map<Node, std::unordered_map<std::string, std::string>>
      node_labels = generate_map(get_nodes(g), get_node_attrs);

  auto get_output_label = [](DataflowOutput const &o) {
    return fmt::to_string(o.idx);
  };

  std::unordered_map<DataflowOutput, std::string> output_labels =
      generate_map(get_all_dataflow_outputs(g), get_output_label);
  std::unordered_map<OpenDataflowValue, std::string> value_labels =
      map_keys(output_labels,
               [](DataflowOutput const &o) { return OpenDataflowValue{o}; });

  return render_dot(with_labelling(
      view_as_open_dataflow_graph(g), node_labels, value_labels));
}

void as_dot(DotFile<std::string> &dot,
            DataflowGraphView const &g,
            std::function<std::string(Node const &)> const &get_node_label) {
  auto get_node_name = [](Node n) { return fmt::format("n{}", n.raw_uid); };

  auto get_input_field = [](nonnegative_int idx) {
    return fmt::format("i{}", idx);
  };

  auto get_output_field = [](nonnegative_int idx) {
    return fmt::format("o{}", idx);
  };

  for (Node const &n : get_nodes(g)) {
    std::vector<DataflowInput> n_inputs = get_dataflow_inputs(g, n);
    std::vector<DataflowOutput> n_outputs = get_outputs(g, n);

    RecordFormatter inputs_record;
    for (DataflowInput const &i : n_inputs) {
      inputs_record << fmt::format("<{}>{}", get_input_field(i.idx), i.idx);
    }

    RecordFormatter outputs_record;
    for (DataflowOutput const &o : n_outputs) {
      outputs_record << fmt::format("<{}>{}", get_output_field(o.idx), o.idx);
    }

    RecordFormatter rec;
    rec << inputs_record << get_node_label(n) << outputs_record;

    dot.add_record_node(get_node_name(n), rec);
  }

  for (DataflowEdge const &e : get_edges(g)) {
    dot.add_edge(get_node_name(e.src.node),
                 get_node_name(e.dst.node),
                 get_output_field(e.src.idx),
                 get_input_field(e.dst.idx));
  }
}

} // namespace FlexFlow
