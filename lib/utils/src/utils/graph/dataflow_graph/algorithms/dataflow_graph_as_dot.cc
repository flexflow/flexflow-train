#include "utils/graph/dataflow_graph/algorithms/dataflow_graph_as_dot.h"
#include "utils/containers/map_keys.h"
#include "utils/dot/dot_file.h"
#include "utils/dot/dot_html_from_json.h"
#include "utils/graph/dataflow_graph/algorithms.h"
#include "utils/graph/dataflow_graph/algorithms/view_as_open_dataflow_graph.h"
#include "utils/graph/labelled_open_dataflow_graph/algorithms/with_labelling.h"
#include "utils/graph/node/algorithms.h"
#include "utils/graph/render_dot.h"
#include "utils/nonnegative_int/num_elements.h"
#include "utils/record_formatter.h"

namespace FlexFlow {

std::string dataflow_graph_as_dot(
    DataflowGraphView const &g,
    std::optional<std::function<nlohmann::json(Node const &)>> const
        &get_node_label,
    std::optional<std::function<nlohmann::json(DataflowInput const &)>> const
        &get_input_label,
    std::optional<std::function<nlohmann::json(DataflowOutput const &)>> const
        &get_output_label) {
  std::ostringstream oss;
  DotFile<std::string> dot{oss};

  dataflow_graph_as_dot(
      dot, g, get_node_label, get_input_label, get_output_label);

  dot.close();

  return oss.str();
}

void dataflow_graph_as_dot(
    DotFile<std::string> &dot,
    DataflowGraphView const &g,
    std::optional<std::function<nlohmann::json(Node const &)>> const
        &get_node_label,
    std::optional<std::function<nlohmann::json(DataflowInput const &)>> const
        &get_input_label,
    std::optional<std::function<nlohmann::json(DataflowOutput const &)>> const
        &get_output_label) {

  auto get_node_name = [](Node n) { return fmt::format("n{}", n.raw_uid); };

  std::function<nlohmann::json(Node const &)> resolved_get_node_label =
      get_node_label.value_or(get_node_name);

  std::function<nlohmann::json(DataflowInput const &)>
      resolved_get_input_label = get_input_label.value_or(
          [](DataflowInput const &i) { return fmt::to_string(i.idx); });

  std::function<nlohmann::json(DataflowOutput const &)>
      resolved_get_output_label = get_output_label.value_or(
          [](DataflowOutput const &o) { return fmt::to_string(o.idx); });

  auto get_input_field = [](nonnegative_int idx) {
    return fmt::format("i{}", idx);
  };

  auto get_output_field = [](nonnegative_int idx) {
    return fmt::format("o{}", idx);
  };

  for (Node const &n : get_nodes(g)) {
    std::vector<DataflowInput> n_inputs = get_dataflow_inputs(g, n);
    std::vector<DataflowOutput> n_outputs = get_outputs(g, n);

    auto make_io_cell = [](nlohmann::json const &j,
                           std::string const &port,
                           positive_int colspan) -> DotHtmlTableCell {
      DotHtmlTableCell cell = dot_html_cell_from_json(j);
      cell.port = port;
      cell.colspan = colspan;
      return cell;
    };

    positive_int num_input_columns =
        positive_int{std::max(num_elements(n_inputs), 1_n)};
    positive_int num_output_columns =
        positive_int{std::max(num_elements(n_outputs), 1_n)};

    std::vector<DotHtmlTableCell> inputs =
        transform(n_inputs, [&](DataflowInput const &i) -> DotHtmlTableCell {
          return make_io_cell(resolved_get_input_label(i),
                              get_input_field(i.idx),
                              num_output_columns);
        });

    if (inputs.size() == 0) {
      inputs.push_back(DotHtmlTableCell{
          /*content=*/DotHtmlTableCellContents{
              "(no inputs)",
          },
          /*port=*/std::nullopt,
          /*colspan=*/num_output_columns,
      });
    }

    DotHtmlTableCell body = dot_html_cell_from_json(resolved_get_node_label(n));
    body.colspan = num_input_columns * num_output_columns;

    std::vector<DotHtmlTableCell> outputs =
        transform(n_outputs, [&](DataflowOutput const &o) -> DotHtmlTableCell {
          return make_io_cell(resolved_get_output_label(o),
                              get_output_field(o.idx),
                              num_input_columns);
        });

    if (outputs.size() == 0) {
      outputs.push_back(DotHtmlTableCell{
          /*content=*/DotHtmlTableCellContents{
              "(no outputs)",
          },
          /*port=*/std::nullopt,
          /*colspan=*/num_input_columns,
      });
    }

    DotHtmlTable table = DotHtmlTable{
        /*border=*/0_n,
        /*cellborder=*/1_n,
        /*cellspacing=*/0_n,
        /*rows=*/
        {
            DotHtmlTableRow{
                inputs,
            },
            DotHtmlTableRow{
                /*cells=*/{
                    body,
                },
            },
            DotHtmlTableRow{
                outputs,
            },
        },
    };

    dot.add_html_node(get_node_name(n), table);
  }

  for (DataflowEdge const &e : get_edges(g)) {
    dot.add_edge(get_node_name(e.src.node),
                 get_node_name(e.dst.node),
                 get_output_field(e.src.idx),
                 get_input_field(e.dst.idx));
  }
}

} // namespace FlexFlow
