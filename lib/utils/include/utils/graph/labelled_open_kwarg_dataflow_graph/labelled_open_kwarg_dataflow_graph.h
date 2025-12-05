#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_OPEN_KWARG_DATAFLOW_GRAPH_LABELLED_OPEN_KWARG_DATAFLOW_GRAPH_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_OPEN_KWARG_DATAFLOW_GRAPH_LABELLED_OPEN_KWARG_DATAFLOW_GRAPH_H

#include "utils/graph/labelled_open_kwarg_dataflow_graph/i_labelled_open_kwarg_dataflow_graph.h"
#include "utils/graph/labelled_open_kwarg_dataflow_graph/labelled_open_kwarg_dataflow_graph_view.h"

namespace FlexFlow {

template <typename NodeLabel, typename ValueLabel, typename GraphInputName, typename SlotName>
struct LabelledOpenKwargDataflowGraph
    : virtual public LabelledOpenKwargDataflowGraphView<NodeLabel, ValueLabel, GraphInputName, SlotName> {
private:
  using Interface = ILabelledOpenKwargDataflowGraph<NodeLabel, ValueLabel, GraphInputName, SlotName>;

public:
  LabelledOpenKwargDataflowGraph(LabelledOpenKwargDataflowGraph const &) = default;
  LabelledOpenKwargDataflowGraph &
      operator=(LabelledOpenKwargDataflowGraph const &) = default;

  KwargNodeAddedResult<SlotName> add_node(
      NodeLabel const &node_label,
      std::unordered_map<SlotName, SingularOrVariadic<OpenKwargDataflowValue<GraphInputName, SlotName>>> const &inputs,
      std::unordered_map<SlotName, SingularOrVariadic<ValueLabel>> const &output_labels) {
    return this->get_interface().add_node(node_label, inputs, output_labels);
  }

  KwargDataflowGraphInput<GraphInputName> add_input(ValueLabel const &value_label) {
    return this->get_interface().add_input(value_label);
  }

  template <typename T>
  static typename std::enable_if<std::is_base_of<Interface, T>::value,
                                 LabelledOpenKwargDataflowGraph>::type
      create() {
    return LabelledOpenKwargDataflowGraph(make_cow_ptr<T>());
  }

  template <typename T>
  static std::enable_if_t<std::is_base_of_v<Interface, T>,
                          LabelledOpenKwargDataflowGraph>
      create_copy_of(
          LabelledOpenKwargDataflowGraphView<NodeLabel, ValueLabel, GraphInputName, SlotName> const &view) {
    cow_ptr_t<T> impl = make_cow_ptr<T>();
    impl.get_mutable()->inplace_materialize_from(view);
    return LabelledOpenKwargDataflowGraph(std::move(impl));
  }

protected:
  using LabelledOpenKwargDataflowGraphView<NodeLabel, ValueLabel, GraphInputName, SlotName>::
      LabelledOpenKwargDataflowGraphView;

private:
  Interface &get_interface() {
    return *std::dynamic_pointer_cast<Interface>(GraphView::ptr.get_mutable());
  }

  Interface const &get_interface() const {
    return *std::dynamic_pointer_cast<Interface const>(GraphView::ptr.get());
  }
};

} // namespace FlexFlow

#endif
