#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_KWARG_DATAFLOW_GRAPH_I_KWARG_DATAFLOW_GRAPH_VIEW_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_KWARG_DATAFLOW_GRAPH_I_KWARG_DATAFLOW_GRAPH_VIEW_H

#include "utils/singular_or_variadic.dtg.h"
#include "utils/graph/dataflow_graph/dataflow_edge.dtg.h"
#include "utils/graph/kwarg_dataflow_graph/kwarg_dataflow_edge.dtg.h"
#include "utils/graph/kwarg_dataflow_graph/kwarg_dataflow_edge_query.dtg.h"
#include "utils/graph/kwarg_dataflow_graph/kwarg_dataflow_output.dtg.h"
#include "utils/graph/kwarg_dataflow_graph/kwarg_dataflow_output_query.dtg.h"
#include "utils/graph/digraph/i_digraph_view.h"

namespace FlexFlow {

template <typename SlotName>
struct IKwargDataflowGraphView : virtual public IDiGraphView {
  virtual std::unordered_set<KwargDataflowEdge<SlotName>>
      query_edges(KwargDataflowEdgeQuery<SlotName> const &) const = 0;
  virtual std::unordered_set<KwargDataflowOutput<SlotName>>
      query_outputs(KwargDataflowOutputQuery<SlotName> const &) const = 0;

  std::unordered_set<DirectedEdge>
      query_edges(DirectedEdgeQuery const &q) const override final {
    KwargDataflowEdgeQuery dataflow_query = KwargDataflowEdgeQuery{
        q.srcs,
        matchall<SlotName>(),
        q.dsts,
        matchall<SlotName>(),
    };
    std::unordered_set<KwargDataflowEdge<SlotName>> dataflow_edges =
        this->query_edges(dataflow_query);

    return transform(dataflow_edges, [](KwargDataflowEdge<SlotName> const &e) {
      return DirectedEdge{e.src.node, e.dst.node};
    });
  };

  virtual ~IKwargDataflowGraphView() = default;

};
CHECK_RC_COPY_VIRTUAL_COMPLIANT(IKwargDataflowGraphView<std::string>);

} // namespace FlexFlow

#endif
