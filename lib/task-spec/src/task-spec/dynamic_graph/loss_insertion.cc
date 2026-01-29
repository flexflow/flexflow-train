#include "pcg/optimizer_attrs.h"
#include "task-spec/dynamic_graph/dynamic_open_dataflow_graph.h"
#include "task-spec/dynamic_graph/dynamic_tensor_role.h"
#include "task-spec/dynamic_graph/dynamic_tensor_slot.h"
#include "task-spec/dynamic_graph/dynamic_value_attrs.h"
#include "task-spec/dynamic_graph/update_insertion.h"
#include "task-spec/optimizer.h"
#include "utils/containers/get_only.h"
#include "utils/containers/map_from_pairs.h"
#include "utils/containers/set_union.h"
#include "utils/exception.h"

namespace FlexFlow {

std::pair<DynamicOpenDataflowGraph, DynamicValueAttrs>
    perform_loss_insertion(DynamicOpenDataflowGraph const &dg,
                           LossAttrs const &loss_attrs,
                           dynamic_tensor_guid_t logit_tensor) {
  NOT_IMPLEMENTED();
}

} // namespace FlexFlow
