#include "utils/graph/render_dot.h"
#include "utils/containers/flatmap.h"
#include "utils/containers/try_at.h"
#include "utils/graph/dataflow_graph/algorithms.h"
#include "utils/graph/node/algorithms.h"
#include "utils/record_formatter.h"

namespace FlexFlow {

std::string escape_dot_string(std::string const &s) {
  auto escape_dot_char = [](char c) -> std::string {
    switch (c) {
      case '\\':
      case '"':
        return std::string{'\\'} + c;
      default:
        return std::string{c};
    }
  };

  return flatmap(s, escape_dot_char);
}

std::string render_dot_node_attrs(
    std::unordered_map<std::string, std::string> const &node_attrs) {
  std::ostringstream oss;
  for (auto const &[k, v] : node_attrs) {
    oss << fmt::format(
        "\"{}\"=\"{}\",", escape_dot_string(k), escape_dot_string(v));
  }
  return oss.str();
}

std::string render_node_label(
    LabelledDataflowGraphView<std::unordered_map<std::string, std::string>,
                              std::string> const &g,
    Node const &n) {
  std::vector<DataflowInput> n_inputs = get_dataflow_inputs(g, n);
  std::vector<DataflowOutput> n_outputs = get_outputs(g, n);

  RecordFormatter inputs_record;
  for (DataflowInput const &i : n_inputs) {
    inputs_record << fmt::format("<i{}>{}", i.idx, i.idx);
  }

  RecordFormatter outputs_record;
  for (DataflowOutput const &o : n_outputs) {
    outputs_record << fmt::format("<o{}>{}", o.idx, g.at(o));
  }

  RecordFormatter rec;
  rec << inputs_record
      << try_at(g.at(n), std::string{"label"})
             .value_or(fmt::to_string(n.raw_uid))
      << outputs_record;

  std::ostringstream oss;
  oss << rec;
  return oss.str();
}

std::string render_dot(
    LabelledDataflowGraphView<std::unordered_map<std::string, std::string>,
                              std::string> const &g) {
  std::vector<std::string> lines;
  lines.push_back("digraph {");

  for (Node const &n : get_nodes(g)) {
    std::unordered_map<std::string, std::string> node_attrs = g.at(n);
    node_attrs.at("label") = render_node_label(g, n);
    node_attrs["shape"] = "record";

    lines.push_back(fmt::format(
        "  n{} [{}];", n.raw_uid, render_dot_node_attrs(node_attrs)));
  }

  for (DataflowEdge const &e : get_edges(g)) {
    lines.push_back(fmt::format("  n{}:o{} -> n{}:i{};",
                                e.src.node.raw_uid,
                                e.src.idx,
                                e.dst.node.raw_uid,
                                e.dst.idx));
  }

  lines.push_back("}");

  return join_strings(lines, "\n");
}

} // namespace FlexFlow
