#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_KWARG_DATAFLOW_GRAPH_I_LABELLED_KWARG_DATAFLOW_GRAPH_VIEW_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_KWARG_DATAFLOW_GRAPH_I_LABELLED_KWARG_DATAFLOW_GRAPH_VIEW_H

#include "utils/graph/kwarg_dataflow_graph/i_kwarg_dataflow_graph_view.h"

namespace FlexFlow {

template <typename NodeLabel,
          typename ValueLabel,
          typename SlotName>
struct ILabelledKwargDataflowGraphView : virtual public IKwargDataflowGraphView<SlotName> {
  virtual NodeLabel at(Node const &) const = 0;
  virtual ValueLabel at(KwargDataflowOutput<SlotName> const &) const = 0;

  virtual ~ILabelledKwargDataflowGraphView() = default;
};
CHECK_RC_COPY_VIRTUAL_COMPLIANT(ILabelledKwargDataflowGraphView<int, std::string, float>);

} // namespace FlexFlow

#endif
