#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_KWARG_DATAFLOW_GRAPH_LABELLED_KWARG_DATAFLOW_GRAPH_VIEW_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_KWARG_DATAFLOW_GRAPH_LABELLED_KWARG_DATAFLOW_GRAPH_VIEW_H

#include "utils/graph/kwarg_dataflow_graph/kwarg_dataflow_graph_view.h"
#include "utils/graph/labelled_kwarg_dataflow_graph/i_labelled_kwarg_dataflow_graph_view.h"

namespace FlexFlow {

template <typename NodeLabel, typename OutputLabel, typename SlotName>
struct LabelledKwargDataflowGraphView
    : virtual public KwargDataflowGraphView<SlotName> {
private:
  using Interface =
      ILabelledKwargDataflowGraphView<NodeLabel, OutputLabel, SlotName>;

public:
  LabelledKwargDataflowGraphView(LabelledKwargDataflowGraphView const &) =
      default;
  LabelledKwargDataflowGraphView &
      operator=(LabelledKwargDataflowGraphView const &) = default;

  NodeLabel at(Node const &n) const {
    return this->get_interface().at(n);
  }

  OutputLabel at(KwargDataflowOutput<SlotName> const &o) const {
    return this->get_interface().at(o);
  }

  template <typename T, typename... Args>
  static typename std::enable_if<std::is_base_of<Interface, T>::value,
                                 LabelledKwargDataflowGraphView>::type
      create(Args &&...args) {
    return LabelledKwargDataflowGraphView(
        make_cow_ptr<T>(std::forward<Args>(args)...));
  }

protected:
  using KwargDataflowGraphView<SlotName>::KwargDataflowGraphView;

private:
  Interface const &get_interface() const {
    return *std::dynamic_pointer_cast<Interface const>(GraphView::ptr.get());
  }
};

} // namespace FlexFlow

#endif
