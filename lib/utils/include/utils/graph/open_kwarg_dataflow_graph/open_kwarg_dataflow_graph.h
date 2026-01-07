#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_OPEN_KWARG_DATAFLOW_GRAPH_OPEN_KWARG_DATAFLOW_GRAPH_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_OPEN_KWARG_DATAFLOW_GRAPH_OPEN_KWARG_DATAFLOW_GRAPH_H

#include "utils/graph/kwarg_dataflow_graph/kwarg_node_added_result.dtg.h"
#include "utils/graph/open_kwarg_dataflow_graph/i_open_kwarg_dataflow_graph.h"
#include "utils/graph/open_kwarg_dataflow_graph/open_kwarg_dataflow_graph_view.h"
#include "utils/graph/open_kwarg_dataflow_graph/open_kwarg_dataflow_value.dtg.h"

namespace FlexFlow {

template <typename GraphInputName, typename SlotName>
struct OpenKwargDataflowGraph
    : virtual public OpenKwargDataflowGraphView<GraphInputName, SlotName> {
public:
  KwargNodeAddedResult<SlotName> add_node(
      std::unordered_map<SlotName,
                         OpenKwargDataflowValue<GraphInputName, SlotName>> const
          &inputs,
      std::unordered_set<SlotName> const &outputs) {
    return this->get_interface().add_node(inputs, outputs);
  }

  KwargDataflowGraphInput<GraphInputName> add_input(GraphInputName const &n) {
    return this->get_interface().add_input(n);
  }

  template <typename T, typename... Args>
  static typename std::enable_if<
      std::is_base_of<IOpenKwargDataflowGraph<GraphInputName, SlotName>,
                      T>::value,
      OpenKwargDataflowGraph>::type
      create(Args &&...args) {
    return OpenKwargDataflowGraph(make_cow_ptr<T>(std::forward<Args>(args)...));
  }

protected:
  using OpenKwargDataflowGraphView<GraphInputName,
                                   SlotName>::OpenKwargDataflowGraphView;

private:
  IOpenKwargDataflowGraph<GraphInputName, SlotName> &get_interface() {
    return *std::dynamic_pointer_cast<
        IOpenKwargDataflowGraph<GraphInputName, SlotName>>(
        GraphView::ptr.get_mutable());
  }

  IOpenKwargDataflowGraph<GraphInputName, SlotName> const &
      get_interface() const {
    return *std::dynamic_pointer_cast<
        IOpenKwargDataflowGraph<GraphInputName, SlotName> const>(
        GraphView::ptr.get());
  }
};

} // namespace FlexFlow

#endif
