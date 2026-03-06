#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_INSTANCES_UNORDERED_SET_KWARG_DATAFLOW_GRAPH_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_INSTANCES_UNORDERED_SET_KWARG_DATAFLOW_GRAPH_H

#include "utils/containers/filter.h"
#include "utils/containers/generate_map.h"
#include "utils/containers/keys.h"
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
public:
  UnorderedSetKwargDataflowGraph() = default;

  KwargNodeAddedResult<SlotName> add_node(
      std::unordered_map<SlotName, KwargDataflowOutput<SlotName>> const &inputs,
      std::unordered_set<SlotName> const &output_slots) override {
    Node new_node = this->node_source.new_node();
    this->nodes.insert(new_node);

    for (auto const &[slot, src] : inputs) {
      this->edges.insert(KwargDataflowEdge<SlotName>{
          src,
          KwargDataflowInput<SlotName>{new_node, slot},
      });
    }

    std::unordered_map<SlotName, KwargDataflowOutput<SlotName>> outputs =
        generate_map(
            output_slots,
            [&](SlotName const &slot) -> KwargDataflowOutput<SlotName> {
              KwargDataflowOutput<SlotName> out{new_node, slot};
              this->outputs.insert(out);
              return out;
            });

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

    for (auto const &[slot, src] : inputs) {
      this->edges.insert(KwargDataflowEdge<SlotName>{
          src,
          KwargDataflowInput<SlotName>{node, slot},
      });
    }

    for (auto const &[slot, out] : outputs) {
      this->outputs.insert(out);
    }
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
    return filter(this->outputs, [&](KwargDataflowOutput<SlotName> const &o) {
      return kwarg_dataflow_output_query_includes(q, o);
    });
  }

  void inplace_materialize_from(
      KwargDataflowGraphView<SlotName> const &view) override {
    this->nodes = get_nodes(view);
    this->edges = get_all_kwarg_dataflow_edges(view);
    this->outputs = get_all_kwarg_dataflow_outputs(view);
  }

  UnorderedSetKwargDataflowGraph *clone() const override {
    return new UnorderedSetKwargDataflowGraph{
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
CHECK_RC_COPY_VIRTUAL_COMPLIANT(UnorderedSetKwargDataflowGraph<int>);

} // namespace FlexFlow

#endif
