#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_OPEN_KWARG_DATAFLOW_GRAPH_I_OPEN_KWARG_DATAFLOW_GRAPH_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_OPEN_KWARG_DATAFLOW_GRAPH_I_OPEN_KWARG_DATAFLOW_GRAPH_H

#include "utils/graph/kwarg_dataflow_graph/kwarg_node_added_result.dtg.h"
#include "utils/graph/open_kwarg_dataflow_graph/i_open_kwarg_dataflow_graph_view.h"
#include "utils/graph/open_kwarg_dataflow_graph/open_kwarg_dataflow_value.dtg.h"

namespace FlexFlow {

template <typename GraphInputName, typename SlotName>
struct IOpenKwargDataflowGraph : virtual public IOpenKwargDataflowGraphView<GraphInputName, SlotName> {
  virtual KwargNodeAddedResult<SlotName> 
    add_node(std::unordered_map<SlotName, OpenKwargDataflowValue<GraphInputName, SlotName>> const &inputs,
             std::unordered_set<SlotName> const &outputs) = 0;
  virtual KwargDataflowGraphInput<GraphInputName> add_input(GraphInputName const &name) = 0;
  virtual IOpenKwargDataflowGraph *clone() const = 0;

  virtual ~IOpenKwargDataflowGraph() = default;
};
CHECK_RC_COPY_VIRTUAL_COMPLIANT(IOpenKwargDataflowGraph<std::string, int>);

} // namespace FlexFlow

#endif
