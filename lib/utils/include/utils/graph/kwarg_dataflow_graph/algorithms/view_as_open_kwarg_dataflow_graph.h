#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_KWARG_DATAFLOW_GRAPH_ALGORITHMS_VIEW_AS_OPEN_KWARG_DATAFLOW_GRAPH_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_KWARG_DATAFLOW_GRAPH_ALGORITHMS_VIEW_AS_OPEN_KWARG_DATAFLOW_GRAPH_H

#include "utils/graph/open_kwarg_dataflow_graph/open_kwarg_dataflow_graph_view.h"

namespace FlexFlow {

template <typename GraphInputName, typename SlotName>
struct KwargDataflowGraphAsOpenView final
    : public IOpenKwargDataflowGraphView<GraphInputName, SlotName> {
public:
  KwargDataflowGraphAsOpenView() = delete;
  KwargDataflowGraphAsOpenView(KwargDataflowGraphView<SlotName> const &g)
      : g(g) {}

  std::unordered_set<Node> query_nodes(NodeQuery const &q) const override {
    return this->g.query_nodes(q);
  }

  std::unordered_set<OpenKwargDataflowEdge<GraphInputName, SlotName>>
      query_edges(OpenKwargDataflowEdgeQuery<GraphInputName, SlotName> const &q)
          const override {
    return transform(this->g.query_edges(q.standard_edge_query),
                     [](KwargDataflowEdge<SlotName> const &e) {
                       return OpenKwargDataflowEdge<GraphInputName, SlotName>{
                           e};
                     });
  }

  std::unordered_set<KwargDataflowOutput<SlotName>> query_outputs(
      KwargDataflowOutputQuery<SlotName> const &q) const override {
    return this->g.query_outputs(q);
  }

  std::unordered_set<KwargDataflowGraphInput<GraphInputName>>
      get_inputs() const override {
    return {};
  }

  KwargDataflowGraphAsOpenView *clone() const override {
    return new KwargDataflowGraphAsOpenView{this->g};
  }

private:
  KwargDataflowGraphView<SlotName> g;
};

template <typename GraphInputName, typename SlotName>
OpenKwargDataflowGraphView<GraphInputName, SlotName>
    view_as_open_kwarg_dataflow_graph(
        KwargDataflowGraphView<SlotName> const &g) {
  return OpenKwargDataflowGraphView<GraphInputName, SlotName>::template create<
      KwargDataflowGraphAsOpenView<GraphInputName, SlotName>>(g);
}

} // namespace FlexFlow

#endif
