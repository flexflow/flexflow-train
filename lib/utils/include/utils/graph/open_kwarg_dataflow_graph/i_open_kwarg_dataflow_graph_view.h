#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_OPEN_KWARG_DATAFLOW_GRAPH_I_OPEN_KWARG_DATAFLOW_GRAPH_VIEW_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_OPEN_KWARG_DATAFLOW_GRAPH_I_OPEN_KWARG_DATAFLOW_GRAPH_VIEW_H

#include "utils/graph/kwarg_dataflow_graph/i_kwarg_dataflow_graph_view.h"
#include "utils/graph/open_kwarg_dataflow_graph/kwarg_dataflow_graph_input.dtg.h"
#include "utils/graph/open_kwarg_dataflow_graph/open_kwarg_dataflow_edge.dtg.h"
#include "utils/graph/open_kwarg_dataflow_graph/open_kwarg_dataflow_edge_query.dtg.h"

namespace FlexFlow {

template <typename GraphInputName, typename SlotName>
struct IOpenKwargDataflowGraphView : virtual public IKwargDataflowGraphView<SlotName> {
  virtual std::unordered_set<KwargDataflowGraphInput<GraphInputName>> get_inputs() const = 0;
  virtual std::unordered_set<OpenKwargDataflowEdge<GraphInputName, SlotName>>
      query_edges(OpenKwargDataflowEdgeQuery<GraphInputName, SlotName> const &) const = 0;

  std::unordered_set<KwargDataflowEdge<SlotName>>
      query_edges(KwargDataflowEdgeQuery<SlotName> const &) const override final {
    NOT_IMPLEMENTED();
  }

  virtual ~IOpenKwargDataflowGraphView() = default;
};
CHECK_RC_COPY_VIRTUAL_COMPLIANT(IOpenKwargDataflowGraphView<std::string, int>);

} // namespace FlexFlow

#endif
