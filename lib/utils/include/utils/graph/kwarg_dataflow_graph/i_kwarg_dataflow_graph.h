#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_KWARG_DATAFLOW_GRAPH_I_KWARG_DATAFLOW_GRAPH_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_KWARG_DATAFLOW_GRAPH_I_KWARG_DATAFLOW_GRAPH_H

#include "utils/graph/kwarg_dataflow_graph/kwarg_dataflow_graph_view.h"
#include "utils/graph/kwarg_dataflow_graph/i_kwarg_dataflow_graph_view.h"
#include "utils/graph/kwarg_dataflow_graph/kwarg_node_added_result.dtg.h"
#include "utils/singular_or_variadic.dtg.h"
#include "utils/slot_num_values.dtg.h"

namespace FlexFlow {

template <typename SlotName>
struct IKwargDataflowGraph : virtual public IKwargDataflowGraphView<SlotName> {
  virtual KwargNodeAddedResult<SlotName> add_node(
      std::unordered_map<SlotName, SingularOrVariadic<KwargDataflowOutput<SlotName>>> const &inputs,
      std::unordered_map<SlotName, SlotNumValues> const &outputs) = 0;

  virtual void add_node_unsafe(Node const &node,
                               std::unordered_map<SlotName, SingularOrVariadic<KwargDataflowOutput<SlotName>>> const &inputs,
                               std::unordered_map<SlotName, SingularOrVariadic<KwargDataflowOutput<SlotName>>> const &outputs) = 0;

  virtual void inplace_materialize_from(KwargDataflowGraphView<SlotName> const &) = 0;

  virtual IKwargDataflowGraph *clone() const = 0;
};
CHECK_RC_COPY_VIRTUAL_COMPLIANT(IKwargDataflowGraph<std::string>);

} // namespace FlexFlow

#endif
