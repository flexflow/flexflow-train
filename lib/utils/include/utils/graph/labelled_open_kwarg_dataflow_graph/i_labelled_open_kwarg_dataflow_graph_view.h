#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_OPEN_KWARG_DATAFLOW_GRAPH_I_LABELLED_OPEN_DATAFLOW_GRAPH_VIEW_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_OPEN_KWARG_DATAFLOW_GRAPH_I_LABELLED_OPEN_DATAFLOW_GRAPH_VIEW_H

#include "utils/graph/labelled_kwarg_dataflow_graph/i_labelled_kwarg_dataflow_graph_view.h"
#include "utils/graph/open_kwarg_dataflow_graph/i_open_kwarg_dataflow_graph_view.h"
#include "utils/graph/open_kwarg_dataflow_graph/open_kwarg_dataflow_value.dtg.h"


namespace FlexFlow {

template <typename NodeLabel,
          typename ValueLabel,
          typename GraphInputName, 
          typename SlotName>
struct ILabelledOpenKwargDataflowGraphView 
  : virtual public ILabelledKwargDataflowGraphView<NodeLabel, ValueLabel, SlotName>,
    virtual public IOpenKwargDataflowGraphView<GraphInputName, SlotName>
{
  virtual NodeLabel at(Node const &) const override = 0;
  virtual ValueLabel at(OpenKwargDataflowValue<GraphInputName, SlotName> const &) const = 0;

  ValueLabel at(KwargDataflowOutput<SlotName> const &o) const override final {
    return this->at(OpenKwargDataflowValue<GraphInputName, SlotName>{o});
  }

  virtual ~ILabelledOpenKwargDataflowGraphView() = default;
};
CHECK_RC_COPY_VIRTUAL_COMPLIANT(ILabelledOpenKwargDataflowGraphView<int, std::string, float, bool>);

} // namespace FlexFlow

#endif
