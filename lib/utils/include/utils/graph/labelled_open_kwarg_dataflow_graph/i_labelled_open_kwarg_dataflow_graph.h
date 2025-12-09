#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_OPEN_KWARG_DATAFLOW_GRAPH_I_LABELLED_OPEN_KWARG_DATAFLOW_GRAPH_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_OPEN_KWARG_DATAFLOW_GRAPH_I_LABELLED_OPEN_KWARG_DATAFLOW_GRAPH_H

#include "utils/graph/kwarg_dataflow_graph/kwarg_node_added_result.dtg.h"
#include "utils/graph/labelled_kwarg_dataflow_graph/i_labelled_kwarg_dataflow_graph.h"
#include "utils/graph/labelled_open_kwarg_dataflow_graph/i_labelled_open_kwarg_dataflow_graph_view.h"
#include "utils/graph/labelled_open_kwarg_dataflow_graph/labelled_open_kwarg_dataflow_graph_view.h"

namespace FlexFlow {

template <typename NodeLabel, typename ValueLabel, typename GraphInputName, typename SlotName>
struct ILabelledOpenKwargDataflowGraph
    : virtual public ILabelledOpenKwargDataflowGraphView<NodeLabel, ValueLabel, GraphInputName, SlotName>,
      virtual public ILabelledKwargDataflowGraphView<NodeLabel, ValueLabel, SlotName> {
  virtual KwargNodeAddedResult<SlotName> add_node(
    NodeLabel const &node_label,
    std::unordered_map<SlotName, OpenKwargDataflowValue<GraphInputName, SlotName>> const &inputs,
    std::unordered_map<SlotName, ValueLabel> const &output_labels) = 0;

  virtual KwargDataflowGraphInput<GraphInputName> add_input(
              GraphInputName const &name, ValueLabel const &value_label) = 0;

  virtual void inplace_materialize_from(
      LabelledOpenKwargDataflowGraphView<NodeLabel, ValueLabel, GraphInputName, SlotName> const &) = 0;

  virtual ~ILabelledOpenKwargDataflowGraph() = default;
};
CHECK_RC_COPY_VIRTUAL_COMPLIANT(ILabelledOpenKwargDataflowGraph<int, int, int, int>);


} // namespace FlexFlow

#endif
