#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_KWARG_DATAFLOW_GRAPH_KWARG_DATAFLOW_GRAPH_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_KWARG_DATAFLOW_GRAPH_KWARG_DATAFLOW_GRAPH_H

#include "utils/graph/kwarg_dataflow_graph/kwarg_dataflow_graph_view.h"
#include "utils/graph/kwarg_dataflow_graph/i_kwarg_dataflow_graph.h"
#include "utils/graph/kwarg_dataflow_graph/kwarg_node_added_result.dtg.h"
#include "utils/singular_or_variadic.dtg.h"
#include "utils/slot_num_values.dtg.h"

namespace FlexFlow {

template <typename SlotName>
struct KwargDataflowGraph : virtual public KwargDataflowGraphView<SlotName> {
public:
  KwargNodeAddedResult<SlotName> 
    add_node(std::unordered_map<SlotName, SingularOrVariadic<KwargDataflowOutput<SlotName>>> const &inputs,
             std::unordered_map<SlotName, SlotNumValues> const &outputs) {
    return this->get_interface().add_node(inputs, outputs);
  }

  void add_node_unsafe(Node const &node,
                       std::unordered_map<SlotName, SingularOrVariadic<KwargDataflowOutput<SlotName>>> const &inputs,
                       std::unordered_map<SlotName, SingularOrVariadic<KwargDataflowOutput<SlotName>>> const &outputs) {
    return this->get_interface().add_node_unsafe(node, inputs, outputs);
  }

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

  template <typename T>
  static typename std::enable_if<std::is_base_of<IKwargDataflowGraph<SlotName>, T>::value,
                                 KwargDataflowGraph<SlotName>>::type
      create() {
    return KwargDataflowGraph(make_cow_ptr<T>());
  }

  template <typename T>
  static std::enable_if_t<std::is_base_of_v<IKwargDataflowGraph<SlotName>, T>, KwargDataflowGraph<SlotName>>
      create_copy_of(KwargDataflowGraphView<SlotName> const &view) {
    cow_ptr_t<T> impl = make_cow_ptr<T>();
    impl.get_mutable()->inplace_materialize_from(view);
    return KwargDataflowGraph(std::move(impl));
  }

protected:
  using KwargDataflowGraphView<SlotName>::KwargDataflowGraphView;

private:
  IKwargDataflowGraph<SlotName> &get_interface() {
    return *std::dynamic_pointer_cast<IKwargDataflowGraph<SlotName>>(
        GraphView::ptr.get_mutable());
  }

  IKwargDataflowGraph<SlotName> const &get_interface() const {
    return *std::dynamic_pointer_cast<IKwargDataflowGraph<SlotName> const>(GraphView::ptr.get());
  }
};

} // namespace FlexFlow

#endif
