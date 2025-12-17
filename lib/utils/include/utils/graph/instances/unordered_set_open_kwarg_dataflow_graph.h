#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_INSTANCES_UNORDERED_SET_OPEN_KWARG_DATAFLOW_GRAPH_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_INSTANCES_UNORDERED_SET_OPEN_KWARG_DATAFLOW_GRAPH_H

#include "utils/graph/node/node_source.h"
#include "utils/graph/open_kwarg_dataflow_graph/i_open_kwarg_dataflow_graph.h"
#include "utils/graph/open_kwarg_dataflow_graph/open_kwarg_dataflow_edge.h"
#include "utils/containers/generate_map.h"
#include "utils/graph/open_kwarg_dataflow_graph/open_kwarg_dataflow_edge_query.h"
#include "utils/graph/kwarg_dataflow_graph/kwarg_dataflow_output_query.h"

namespace FlexFlow {

template <typename GraphInputName,
          typename SlotName>
struct UnorderedSetOpenKwargDataflowGraph final
  : public IOpenKwargDataflowGraph<GraphInputName, SlotName>
{
  UnorderedSetOpenKwargDataflowGraph() = default;

  KwargNodeAddedResult<SlotName> 
    add_node(std::unordered_map<SlotName, OpenKwargDataflowValue<GraphInputName, SlotName>> const &inputs,
             std::unordered_set<SlotName> const &output_slots) override {
    Node new_node = this->node_source.new_node();
    this->nodes.insert(new_node);

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
      generate_map(output_slots,
                   [&](SlotName const &output_slot) 
                     -> KwargDataflowOutput<SlotName>
                   {
                     KwargDataflowOutput<SlotName> output = KwargDataflowOutput<SlotName>{
                       /*node=*/new_node,
                       /*slot_name=*/output_slot,
                     };
                     
                     this->outputs.insert(output);

                     return output;
                   });

    return KwargNodeAddedResult<SlotName>{
      /*node=*/new_node,
      /*outputs=*/outputs,
    };

  }

  KwargDataflowGraphInput<GraphInputName> add_input(GraphInputName const &name) override {
    KwargDataflowGraphInput<GraphInputName> input 
      = KwargDataflowGraphInput{name};

    this->graph_inputs.insert(input);

    return input;
  }

  std::unordered_set<Node> query_nodes(NodeQuery const &q) const override {
    return filter(this->nodes,
                  [&](Node const &n) { return includes(q.nodes, n); });
  }

  std::unordered_set<OpenKwargDataflowEdge<GraphInputName, SlotName>>
      query_edges(OpenKwargDataflowEdgeQuery<GraphInputName, SlotName> const &q) const override {
    return filter(this->edges, 
                  [&](OpenKwargDataflowEdge<GraphInputName, SlotName> const &e) {
                    return open_kwarg_dataflow_edge_query_includes(q, e); 
                  });
  }

  std::unordered_set<KwargDataflowOutput<SlotName>>
      query_outputs(KwargDataflowOutputQuery<SlotName> const &q) const override {
    return filter(this->outputs, 
                  [&](KwargDataflowOutput<SlotName> const &output) {
                    return kwarg_dataflow_output_query_includes(q, output); 
                  });
  }

  std::unordered_set<KwargDataflowGraphInput<GraphInputName>> get_inputs() const override {
    return this->graph_inputs;
  }

  UnorderedSetOpenKwargDataflowGraph *clone() const override {
    return new UnorderedSetOpenKwargDataflowGraph<GraphInputName, SlotName>{
      this->node_source,
      this->graph_inputs, 
      this->nodes,
      this->edges,
      this->outputs,
    };
  }

private:
  UnorderedSetOpenKwargDataflowGraph(
    NodeSource const &node_source,
    std::unordered_set<KwargDataflowGraphInput<GraphInputName>> const &graph_inputs,
    std::unordered_set<Node> const &nodes,
    std::unordered_set<OpenKwargDataflowEdge<GraphInputName, SlotName>> const &edges,
    std::unordered_set<KwargDataflowOutput<SlotName>> const &outputs)
    : node_source(node_source),
      graph_inputs(graph_inputs),
      nodes(nodes),
      edges(edges),
      outputs(outputs)
    { }
private:
  NodeSource node_source;

  std::unordered_set<KwargDataflowGraphInput<GraphInputName>> graph_inputs;
  std::unordered_set<Node> nodes;
  std::unordered_set<OpenKwargDataflowEdge<GraphInputName, SlotName>> edges;
  std::unordered_set<KwargDataflowOutput<SlotName>> outputs;
};

} // namespace FlexFlow

#endif
