#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_KWARG_DATAFLOW_GRAPH_ALGORITHMS_VIEW_AS_LABELLED_OPEN_KWARG_DATAFLOW_GRAPH_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_KWARG_DATAFLOW_GRAPH_ALGORITHMS_VIEW_AS_LABELLED_OPEN_KWARG_DATAFLOW_GRAPH_H

#include "utils/graph/labelled_open_kwarg_dataflow_graph/i_labelled_open_kwarg_dataflow_graph_view.h"
#include "utils/graph/labelled_kwarg_dataflow_graph/labelled_kwarg_dataflow_graph_view.h"
#include "utils/graph/labelled_open_kwarg_dataflow_graph/labelled_open_kwarg_dataflow_graph_view.h"

namespace FlexFlow {

template <typename NodeLabel, typename ValueLabel, typename GraphInputName, typename SlotName>
struct LabelledKwargDataflowGraphAsOpenView final 
    : public ILabelledOpenKwargDataflowGraphView<NodeLabel, ValueLabel, GraphInputName, SlotName> 
{
public:
  LabelledKwargDataflowGraphAsOpenView() = delete;
  LabelledKwargDataflowGraphAsOpenView(
    LabelledKwargDataflowGraphView<NodeLabel, ValueLabel, SlotName> const &g)
      : g(g) { }

  std::unordered_set<Node> query_nodes(NodeQuery const &q) const override {
    return this->g.query_nodes(q);
  }

  std::unordered_set<OpenKwargDataflowEdge<GraphInputName, SlotName>>
    query_edges(OpenKwargDataflowEdgeQuery<GraphInputName, SlotName> const &q) const override 
  {
    return transform(this->g.query_edges(q.standard_edge_query),
                     [](KwargDataflowEdge<SlotName> const &e) { 
                       return OpenKwargDataflowEdge<GraphInputName, SlotName>{e};
                     });
  }

  std::unordered_set<KwargDataflowOutput<SlotName>>
      query_outputs(KwargDataflowOutputQuery<SlotName> const &q) const override {
    return this->g.query_outputs(q);
  }

  std::unordered_set<DataflowGraphInput> get_inputs() const override {
    return {};
  }

  NodeLabel at(Node const &n) const override {
    return this->g.at(n);
  }

  ValueLabel at(OpenKwargDataflowValue<GraphInputName, SlotName> const &v) const override {
    return this->g.at(v.require_internal());
  }

  LabelledKwargDataflowGraphAsOpenView *clone() const override {
    return new LabelledKwargDataflowGraphAsOpenView{this->g};
  }
  
private:
  LabelledKwargDataflowGraphView<NodeLabel, ValueLabel, SlotName> g;
};

template <typename NodeLabel, typename ValueLabel, typename GraphInputName, typename SlotName>
LabelledOpenKwargDataflowGraphView<NodeLabel, ValueLabel, GraphInputName, SlotName>
    view_as_labelled_open_kwarg_dataflow_graph(
        LabelledKwargDataflowGraphView<NodeLabel, ValueLabel, SlotName> const &g)
{
  return LabelledOpenKwargDataflowGraphView<NodeLabel, ValueLabel, GraphInputName, SlotName>::template create<
      LabelledKwargDataflowGraphAsOpenView<NodeLabel, ValueLabel, GraphInputName, SlotName>>(g);
}

} // namespace FlexFlow

#endif
