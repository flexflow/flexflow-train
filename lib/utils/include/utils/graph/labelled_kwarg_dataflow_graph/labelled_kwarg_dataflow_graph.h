#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_KWARG_DATAFLOW_GRAPH_LABELLED_KWARG_DATAFLOW_GRAPH_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_KWARG_DATAFLOW_GRAPH_LABELLED_KWARG_DATAFLOW_GRAPH_H

#include "utils/graph/labelled_kwarg_dataflow_graph/i_labelled_kwarg_dataflow_graph.h"
#include "utils/graph/labelled_kwarg_dataflow_graph/labelled_kwarg_dataflow_graph_view.h"

namespace FlexFlow {

template <typename NodeLabel, typename OutputLabel, typename SlotName>
struct LabelledKwargDataflowGraph
    : virtual LabelledKwargDataflowGraphView<NodeLabel, OutputLabel, SlotName> {
private:
  using Interface = ILabelledKwargDataflowGraph<NodeLabel, OutputLabel, SlotName>;

public:
  LabelledKwargDataflowGraph(LabelledKwargDataflowGraph const &) = default;
  LabelledKwargDataflowGraph &operator=(LabelledKwargDataflowGraph const &) = default;

  KwargNodeAddedResult<SlotName> add_node(
      NodeLabel const &node_label,
      std::unordered_map<SlotName, SingularOrVariadic<KwargDataflowOutput<SlotName>>> const &inputs,
      std::unordered_map<SlotName, SingularOrVariadic<OutputLabel>> const &output_labels) {
    return this->get_interface().add_node(node_label, inputs, output_labels);
  }

  template <typename T, typename... Args>
  static typename std::enable_if<std::is_base_of<Interface, T>::value,
                                 LabelledKwargDataflowGraph>::type
      create(Args &&...args) {
    return LabelledKwargDataflowGraph(make_cow_ptr<T>(std::forward<Args>(args)...));
  }

  template <typename T>
  static typename std::enable_if<std::is_base_of<Interface, T>::value,
                                 LabelledKwargDataflowGraph>::type
      create_copy_of(
          LabelledKwargDataflowGraphView<NodeLabel, OutputLabel, SlotName> const &view) {
    cow_ptr_t<T> impl = make_cow_ptr<T>();
    impl.get_mutable()->inplace_materialize_from(view);
    return LabelledKwargDataflowGraph(std::move(impl));
  }

protected:
  using LabelledKwargDataflowGraphView<NodeLabel, OutputLabel, SlotName>::LabelledKwargDataflowGraphView;

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
