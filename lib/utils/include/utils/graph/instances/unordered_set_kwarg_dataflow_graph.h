#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_INSTANCES_UNORDERED_SET_KWARG_DATAFLOW_GRAPH_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_INSTANCES_UNORDERED_SET_KWARG_DATAFLOW_GRAPH_H

#include "utils/containers/generate_unordered_map.h"
#include "utils/containers/set_union.h"
#include "utils/containers/values.h"
#include "utils/graph/kwarg_dataflow_graph/algorithms/get_all_kwarg_dataflow_edges.h"
#include "utils/graph/kwarg_dataflow_graph/algorithms/get_all_kwarg_dataflow_outputs.h"
#include "utils/graph/kwarg_dataflow_graph/i_kwarg_dataflow_graph.h"
#include "utils/graph/kwarg_dataflow_graph/kwarg_dataflow_edge_query.h"
#include "utils/graph/kwarg_dataflow_graph/kwarg_dataflow_output_query.h"
#include "utils/graph/node/algorithms.h"
#include "utils/graph/node/node_source.h"

namespace FlexFlow {

template <typename SlotName>
struct UnorderedSetKwargDataflowGraph final
    : public IKwargDataflowGraph<SlotName> {
  UnorderedSetKwargDataflowGraph() = default;

  KwargNodeAddedResult<SlotName> add_node(
      std::unordered_map<SlotName, KwargDataflowOutput<SlotName>> const &inputs,
      std::unordered_set<SlotName> const &output_slots) override {

    Node new_node = this->node_source.new_node();

    std::unordered_map<SlotName, KwargDataflowOutput<SlotName>> outputs =
        generate_unordered_map(
            output_slots,
            [&](SlotName const &output_slot) -> KwargDataflowOutput<SlotName> {
              KwargDataflowOutput<SlotName> output =
                  KwargDataflowOutput<SlotName>{
                      /*node=*/new_node,
                      /*slot_name=*/output_slot,
                  };

              this->outputs.insert(output);

              return output;
            });

    this->add_node_unsafe(new_node, inputs, outputs);

    return KwargNodeAddedResult<SlotName>{
        /*node=*/new_node,
        /*outputs=*/outputs,
    };
  }

  void add_node_unsafe(
      Node const &node,
      std::unordered_map<SlotName, KwargDataflowOutput<SlotName>> const &inputs,
      std::unordered_map<SlotName, KwargDataflowOutput<SlotName>> const
          &outputs) override {
    this->nodes.insert(node);

    for (auto const &[input_slot_name, src] : inputs) {
      KwargDataflowInput<SlotName> dst = KwargDataflowInput<SlotName>{
          node,
          input_slot_name,
      };

      KwargDataflowEdge<SlotName> in_edge = KwargDataflowEdge{
          /*src=*/src,
          /*dst=*/dst,
      };

      this->edges.insert(in_edge);
    }

    this->outputs = set_union(this->outputs, unordered_set_of(values(outputs)));
  }

  std::unordered_set<Node> query_nodes(NodeQuery const &q) const override {
    return filter(this->nodes,
                  [&](Node const &n) { return includes(q.nodes, n); });
  }

  std::unordered_set<KwargDataflowEdge<SlotName>>
      query_edges(KwargDataflowEdgeQuery<SlotName> const &q) const override {
    return filter(this->edges, [&](KwargDataflowEdge<SlotName> const &e) {
      return kwarg_dataflow_edge_query_includes(q, e);
    });
  }

  std::unordered_set<KwargDataflowOutput<SlotName>> query_outputs(
      KwargDataflowOutputQuery<SlotName> const &q) const override {
    return filter(this->outputs,
                  [&](KwargDataflowOutput<SlotName> const &output) {
                    return kwarg_dataflow_output_query_includes(q, output);
                  });
  }

  void inplace_materialize_from(
      KwargDataflowGraphView<SlotName> const &v) override {
    this->nodes = get_nodes(v);
    this->edges = get_all_kwarg_dataflow_edges(v);
    this->outputs = get_all_kwarg_dataflow_outputs(v);
  };

  UnorderedSetKwargDataflowGraph *clone() const override {
    return new UnorderedSetKwargDataflowGraph<SlotName>{
        this->node_source,
        this->nodes,
        this->edges,
        this->outputs,
    };
  }

private:
  UnorderedSetKwargDataflowGraph(
      NodeSource const &node_source,
      std::unordered_set<Node> const &nodes,
      std::unordered_set<KwargDataflowEdge<SlotName>> const &edges,
      std::unordered_set<KwargDataflowOutput<SlotName>> const &outputs)
      : node_source(node_source), nodes(nodes), edges(edges), outputs(outputs) {
  }

private:
  NodeSource node_source;

  std::unordered_set<Node> nodes;
  std::unordered_set<KwargDataflowEdge<SlotName>> edges;
  std::unordered_set<KwargDataflowOutput<SlotName>> outputs;
};

} // namespace FlexFlow

#endif
