#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_INSTANCES_TASK_SET_OPEN_KWARG_DATAFLOW_GRAPH_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_INSTANCES_TASK_SET_OPEN_KWARG_DATAFLOW_GRAPH_H

#include "utils/containers/contains_key.h"
#include "utils/containers/enumerate.h"
#include "utils/containers/extend.h"
#include "utils/containers/generate_map.h"
#include "utils/containers/map_values.h"
#include "utils/graph/kwarg_dataflow_graph/algorithms/get_all_kwarg_dataflow_edges.h"
#include "utils/graph/kwarg_dataflow_graph/algorithms/get_all_kwarg_dataflow_outputs.h"
#include "utils/graph/kwarg_dataflow_graph/kwarg_node_added_result.dtg.h"
#include "utils/graph/labelled_open_kwarg_dataflow_graph/i_labelled_open_kwarg_dataflow_graph.h"
#include "utils/graph/labelled_open_kwarg_dataflow_graph/i_labelled_open_kwarg_dataflow_graph_view.h"
#include "utils/graph/node/algorithms.h"
#include "utils/graph/node/node_source.h"
#include "utils/graph/open_kwarg_dataflow_graph/algorithms/get_all_kwarg_dataflow_graph_inputs.h"
#include "utils/graph/open_kwarg_dataflow_graph/algorithms/get_all_open_kwarg_dataflow_edges.h"
#include "utils/graph/open_kwarg_dataflow_graph/open_kwarg_dataflow_edge.h"
#include "utils/overload.h"

namespace FlexFlow {

template <typename NodeLabel,
          typename ValueLabel,
          typename GraphInputName,
          typename SlotName>
struct UnorderedSetLabelledOpenKwargDataflowGraph final
    : public ILabelledOpenKwargDataflowGraph<NodeLabel,
                                             ValueLabel,
                                             GraphInputName,
                                             SlotName>,
      public ILabelledKwargDataflowGraph<NodeLabel, ValueLabel, SlotName> {
public:
  UnorderedSetLabelledOpenKwargDataflowGraph() = default;

  KwargNodeAddedResult<SlotName> add_node(
      NodeLabel const &node_label,
      std::unordered_map<SlotName, KwargDataflowOutput<SlotName>> const &inputs,
      std::unordered_map<SlotName, ValueLabel> const &output_labels) override {
    return this->add_node(
        node_label,
        map_values(inputs,
                   [](KwargDataflowOutput<SlotName> const &o) {
                     return OpenKwargDataflowValue<GraphInputName, SlotName>{o};
                   }),
        output_labels);
  };

  KwargNodeAddedResult<SlotName> add_node(
      NodeLabel const &node_label,
      std::unordered_map<SlotName,
                         OpenKwargDataflowValue<GraphInputName, SlotName>> const
          &inputs,
      std::unordered_map<SlotName, ValueLabel> const &output_labels) override {
    Node new_node = this->node_source.new_node();
    this->nodes.insert({new_node, node_label});

    for (auto const &[input_slot_name, input_val] : inputs) {
      KwargDataflowInput<SlotName> dst = KwargDataflowInput<SlotName>{
          new_node,
          input_slot_name,
      };

      OpenKwargDataflowEdge<GraphInputName, SlotName> in_edge =
          mk_open_kwarg_dataflow_edge_from_src_val_and_dst(input_val, dst);

      this->edges.insert(in_edge);
    }

    std::unordered_map<SlotName, KwargDataflowOutput<SlotName>> outputs =
        generate_map(
            keys(output_labels),
            [&](SlotName const &output_slot) -> KwargDataflowOutput<SlotName> {
              ValueLabel value_label = output_labels.at(output_slot);

              KwargDataflowOutput<SlotName> output =
                  KwargDataflowOutput<SlotName>{
                      /*node=*/new_node,
                      /*slot_name=*/output_slot,
                  };

              this->outputs.insert({
                  output,
                  value_label,
              });

              return output;
            });

    return KwargNodeAddedResult<SlotName>{
        /*node=*/new_node,
        /*outputs=*/outputs,
    };
  }

  KwargDataflowGraphInput<GraphInputName>
      add_input(GraphInputName const &name,
                ValueLabel const &value_label) override {
    KwargDataflowGraphInput<GraphInputName> input =
        KwargDataflowGraphInput{name};

    ASSERT(!contains_key(this->graph_inputs, input));
    this->graph_inputs.insert({input, value_label});

    return input;
  }

  std::unordered_set<Node> query_nodes(NodeQuery const &q) const override {
    return filter(keys(this->nodes),
                  [&](Node const &n) { return includes(q.nodes, n); });
  }

  std::unordered_set<OpenKwargDataflowEdge<GraphInputName, SlotName>>
      query_edges(OpenKwargDataflowEdgeQuery<GraphInputName, SlotName> const &q)
          const override {
    return filter(
        this->edges,
        [&](OpenKwargDataflowEdge<GraphInputName, SlotName> const &e) {
          return open_kwarg_dataflow_edge_query_includes(q, e);
        });
  }

  std::unordered_set<KwargDataflowOutput<SlotName>> query_outputs(
      KwargDataflowOutputQuery<SlotName> const &q) const override {
    return filter(keys(this->outputs),
                  [&](KwargDataflowOutput<SlotName> const &output) {
                    return kwarg_dataflow_output_query_includes(q, output);
                  });
  }

  std::unordered_set<KwargDataflowGraphInput<GraphInputName>>
      get_inputs() const override {
    return keys(this->graph_inputs);
  }

  NodeLabel at(Node const &n) const override {
    return this->nodes.at(n);
  }

  ValueLabel at(OpenKwargDataflowValue<GraphInputName, SlotName> const &v)
      const override {
    return v.template visit<ValueLabel>(overload{
        [&](KwargDataflowOutput<SlotName> const &o) -> ValueLabel {
          return this->outputs.at(o);
        },
        [&](KwargDataflowGraphInput<GraphInputName> const &gi) -> ValueLabel {
          return this->graph_inputs.at(gi);
        }});
  }

  void inplace_materialize_from(
      LabelledKwargDataflowGraphView<NodeLabel, ValueLabel, SlotName> const
          &view) override {
    std::unordered_set<Node> view_nodes = get_nodes(view);
    std::unordered_set<KwargDataflowEdge<SlotName>> view_edges =
        get_all_kwarg_dataflow_edges(view);
    std::unordered_set<KwargDataflowOutput<SlotName>> view_outputs =
        get_all_kwarg_dataflow_outputs(view);

    this->graph_inputs.clear();
    this->nodes =
        generate_map(view_nodes, [&](Node const &n) { return view.at(n); });

    this->edges =
        transform(view_edges,
                  [&](KwargDataflowEdge<SlotName> const &e)
                      -> OpenKwargDataflowEdge<GraphInputName, SlotName> {
                    return OpenKwargDataflowEdge<GraphInputName, SlotName>{e};
                  });
    this->outputs =
        generate_map(view_outputs, [&](KwargDataflowOutput<SlotName> const &o) {
          return view.at(o);
        });
  }

  void inplace_materialize_from(
      LabelledOpenKwargDataflowGraphView<NodeLabel,
                                         ValueLabel,
                                         GraphInputName,
                                         SlotName> const &view) override {
    std::unordered_set<KwargDataflowGraphInput<GraphInputName>> view_inputs =
        get_all_kwarg_dataflow_graph_inputs(view);
    std::unordered_set<Node> view_nodes = get_nodes(view);
    std::unordered_set<OpenKwargDataflowEdge<GraphInputName, SlotName>>
        view_edges = get_all_open_kwarg_dataflow_edges(view);
    std::unordered_set<KwargDataflowOutput<SlotName>> view_outputs =
        get_all_kwarg_dataflow_outputs(view);

    this->graph_inputs = generate_map(
        view_inputs, [&](KwargDataflowGraphInput<GraphInputName> const &i) {
          return view.at(OpenKwargDataflowValue<GraphInputName, SlotName>{i});
        });
    this->nodes =
        generate_map(view_nodes, [&](Node const &n) { return view.at(n); });

    this->edges = view_edges;
    this->outputs =
        generate_map(view_outputs, [&](KwargDataflowOutput<SlotName> const &o) {
          return view.at(OpenKwargDataflowValue<GraphInputName, SlotName>{o});
        });
  }

  UnorderedSetLabelledOpenKwargDataflowGraph *clone() const override {
    return new UnorderedSetLabelledOpenKwargDataflowGraph{
        this->node_source,
        this->graph_inputs,
        this->nodes,
        this->edges,
        this->outputs,
    };
  }

private:
  UnorderedSetLabelledOpenKwargDataflowGraph(
      NodeSource const &node_source,
      std::unordered_map<KwargDataflowGraphInput<GraphInputName>,
                         ValueLabel> const &graph_inputs,
      std::unordered_map<Node, NodeLabel> const &nodes,
      std::unordered_set<OpenKwargDataflowEdge<GraphInputName, SlotName>> const
          &edges,
      std::unordered_map<KwargDataflowOutput<SlotName>, ValueLabel> const
          &outputs)
      : node_source(node_source), graph_inputs(graph_inputs), nodes(nodes),
        edges(edges), outputs(outputs) {}

private:
  NodeSource node_source;

  std::unordered_map<KwargDataflowGraphInput<GraphInputName>, ValueLabel>
      graph_inputs;
  std::unordered_map<Node, NodeLabel> nodes;
  std::unordered_set<OpenKwargDataflowEdge<GraphInputName, SlotName>> edges;
  std::unordered_map<KwargDataflowOutput<SlotName>, ValueLabel> outputs;
};

} // namespace FlexFlow

#endif
