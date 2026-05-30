#include "task-spec/dynamic_graph/dynamic_graph_edge.h"

namespace FlexFlow {

DynamicGraphEdge
    dynamic_graph_edge_from_slot_sites(DynamicSlotSite const &src,
                                       InternalDynamicSlotSite const &dst) {
  if (src.is_internal()) {
    ASSERT(src.require_internal().direction == TensorDirection::OUTPUT);
  }
  ASSERT(dst.direction == TensorDirection::INCOMING);

  return DynamicGraphEdge{
      /*src=*/src,
      /*dst_node=*/dst.invocation,
      /*dst_slot=*/dst.slot_name,
  };
}

} // namespace FlexFlow
