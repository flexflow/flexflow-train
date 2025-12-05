#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_OPEN_KWARG_DATAFLOW_GRAPH_LABELLED_OPEN_KWARG_DATAFLOW_GRAPH_VIEW_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_OPEN_KWARG_DATAFLOW_GRAPH_LABELLED_OPEN_KWARG_DATAFLOW_GRAPH_VIEW_H

#include "utils/graph/labelled_kwarg_dataflow_graph/labelled_kwarg_dataflow_graph_view.h"
#include "utils/graph/labelled_open_kwarg_dataflow_graph/i_labelled_open_kwarg_dataflow_graph_view.h"
#include "utils/graph/open_kwarg_dataflow_graph/open_kwarg_dataflow_graph_view.h"

namespace FlexFlow {

template <typename NodeLabel, typename ValueLabel, typename GraphInputName, typename SlotName>
struct LabelledOpenKwargDataflowGraphView
    : virtual public LabelledKwargDataflowGraphView<NodeLabel, ValueLabel, SlotName>,
      virtual public OpenKwargDataflowGraphView<GraphInputName, SlotName> {
private:
  using Interface = ILabelledOpenKwargDataflowGraphView<NodeLabel, ValueLabel, GraphInputName, SlotName>;

public:
  LabelledOpenKwargDataflowGraphView(LabelledOpenKwargDataflowGraphView const &) =
      default;
  LabelledOpenKwargDataflowGraphView &
      operator=(LabelledOpenKwargDataflowGraphView const &) = default;

  NodeLabel at(Node const &n) const {
    return this->get_interface().at(n);
  }

  ValueLabel at(OpenKwargDataflowValue<GraphInputName, SlotName> const &v) const {
    return this->get_interface().at(v);
  }

  template <typename T, typename... Args>
  static typename std::enable_if<
      std::is_base_of<Interface, T>::value,
      LabelledOpenKwargDataflowGraphView<NodeLabel, ValueLabel, GraphInputName, SlotName>>::type
      create(Args &&...args) {
    return LabelledOpenKwargDataflowGraphView(static_cast<cow_ptr_t<IGraphView>>(
        make_cow_ptr<T>(std::forward<Args>(args)...)));
  }

protected:
  using OpenKwargDataflowGraphView<GraphInputName, SlotName>::OpenKwargDataflowGraphView;

private:
  Interface const &get_interface() const {
    return *std::dynamic_pointer_cast<Interface const>(GraphView::ptr.get());
  }
};


} // namespace FlexFlow

#endif
