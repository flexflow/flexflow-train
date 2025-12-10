#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_OPEN_KWARG_DATAFLOW_GRAPH_OPEN_KWARG_DATAFLOW_GRAPH_VIEW_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_OPEN_KWARG_DATAFLOW_GRAPH_OPEN_KWARG_DATAFLOW_GRAPH_VIEW_H

#include "utils/graph/kwarg_dataflow_graph/kwarg_dataflow_graph_view.h"
#include "utils/graph/open_kwarg_dataflow_graph/i_open_kwarg_dataflow_graph_view.h"

namespace FlexFlow {

template <typename GraphInputName, typename SlotName>
struct OpenKwargDataflowGraphView : virtual public KwargDataflowGraphView<SlotName> {
public:
  OpenKwargDataflowGraphView(OpenKwargDataflowGraphView const &) = default;
  OpenKwargDataflowGraphView &operator=(OpenKwargDataflowGraphView const &) = default;

  std::unordered_set<KwargDataflowGraphInput<GraphInputName>> get_inputs() const {
    return this->get_interface().get_inputs();
  }

  std::unordered_set<OpenKwargDataflowEdge<GraphInputName, SlotName>>
      query_edges(OpenKwargDataflowEdgeQuery<GraphInputName, SlotName> const &q) const {
    return this->get_interface().query_edges(q);
  }

  template <typename T, typename... Args>
  static
      typename std::enable_if<std::is_base_of<IOpenKwargDataflowGraphView<GraphInputName, SlotName>, T>::value,
                              OpenKwargDataflowGraphView>::type
      create(Args &&...args) {
    return OpenKwargDataflowGraphView(make_cow_ptr<T>(std::forward<Args>(args)...));
  }

protected:
  using KwargDataflowGraphView<SlotName>::KwargDataflowGraphView;

private:
  IOpenKwargDataflowGraphView<GraphInputName, SlotName> const &get_interface() const {
  return *std::dynamic_pointer_cast<IOpenKwargDataflowGraphView<GraphInputName, SlotName> const>(
      GraphView::ptr.get());
  }
};

} // namespace FlexFlow

#endif
