#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_KWARG_DATAFLOW_GRAPH_KWARG_DATAFLOW_GRAPH_VIEW_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_KWARG_DATAFLOW_GRAPH_KWARG_DATAFLOW_GRAPH_VIEW_H

#include "utils/singular_or_variadic.dtg.h"
#include "utils/graph/kwarg_dataflow_graph/kwarg_dataflow_edge_query.dtg.h"
#include "utils/graph/kwarg_dataflow_graph/kwarg_dataflow_output_query.dtg.h"
#include "utils/graph/kwarg_dataflow_graph/i_kwarg_dataflow_graph_view.h"
#include "utils/graph/digraph/digraph_view.h"

namespace FlexFlow {

template <typename SlotName>
struct KwargDataflowGraphView : virtual public DiGraphView {
  KwargDataflowGraphView(KwargDataflowGraphView const &) = default;
  KwargDataflowGraphView &operator=(KwargDataflowGraphView const &) = default;

  std::unordered_set<Node> query_nodes(NodeQuery const &q) const {
    return this->get_interface().query_nodes(q);
  }

  std::unordered_set<KwargDataflowEdge<SlotName>> query_edges(KwargDataflowEdgeQuery<SlotName> const &q) const {
    return this->get_interface().query_edges(q);
  }

  std::unordered_set<KwargDataflowOutput<SlotName>>
      query_outputs(KwargDataflowOutputQuery<SlotName> const &q) const {
    return this->get_interface().query_outputs(q);
  }

  template <typename T, typename... Args>
  static typename std::enable_if<std::is_base_of<IKwargDataflowGraphView<SlotName>, T>::value,
                                 KwargDataflowGraphView<SlotName>>::type
      create(Args &&...args) {
    return DataflowGraphView(make_cow_ptr<T>(std::forward<Args>(args)...));
  }

protected:
  using DiGraphView::DiGraphView;

private:
  IKwargDataflowGraphView<SlotName> const &get_interface() const {
    return *std::dynamic_pointer_cast<IKwargDataflowGraphView<SlotName> const>(
        GraphView::ptr.get());
  }
};

} // namespace FlexFlow

#endif
