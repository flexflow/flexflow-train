#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_KWARG_DATAFLOW_GRAPH_I_LABELLED_KWARG_DATAFLOW_GRAPH_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_KWARG_DATAFLOW_GRAPH_I_LABELLED_KWARG_DATAFLOW_GRAPH_H

#include "utils/graph/kwarg_dataflow_graph/kwarg_node_added_result.dtg.h"
#include "utils/graph/labelled_kwarg_dataflow_graph/i_labelled_kwarg_dataflow_graph_view.h"
#include "utils/graph/labelled_kwarg_dataflow_graph/labelled_kwarg_dataflow_graph_view.h"

namespace FlexFlow {

template <typename NodeLabel, typename OutputLabel, typename SlotName>
struct ILabelledKwargDataflowGraph
  : virtual public ILabelledKwargDataflowGraphView<NodeLabel, OutputLabel, SlotName> {
public:
  virtual KwargNodeAddedResult<SlotName>
      add_node(NodeLabel const &node_label,
               std::unordered_map<SlotName, SingularOrVariadic<KwargDataflowOutput<SlotName>>> const &inputs,
               std::unordered_map<SlotName, SingularOrVariadic<OutputLabel>> const &output_labels) = 0;
virtual void inplace_materialize_from(
      LabelledKwargDataflowGraphView<NodeLabel, OutputLabel, SlotName> const &) = 0;

  virtual ~ILabelledKwargDataflowGraph() = default;
};

} // namespace FlexFlow

#endif
