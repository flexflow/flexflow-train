#include "pcg/optimizer_attrs.h"
#include "task-spec/dynamic_graph/dynamic_node_invocation.dtg.h"
#include "task-spec/dynamic_graph/dynamic_open_dataflow_graph.h"
#include "task-spec/dynamic_graph/dynamic_tensor_role.h"
#include "task-spec/dynamic_graph/dynamic_tensor_slot.h"
#include "task-spec/dynamic_graph/dynamic_value_attrs.dtg.h"
#include "task-spec/dynamic_graph/dynamic_value_attrs.h"
#include "task-spec/optimizer.h"
#include "utils/containers/get_only.h"
#include "utils/containers/map_from_pairs.h"
#include "utils/containers/set_union.h"
#include "utils/exception.h"
#include "utils/optional.h"
#include <optional>

namespace FlexFlow {

static std::optional<std::pair<DynamicNodeInvocation, DynamicValueAttrs>>
    find_output_tensor(DynamicOpenDataflowGraph const &dg,
                       dynamic_tensor_guid_t tensor_guid) {
  for (DynamicNodeInvocation const &invocation : dg.invocations) {
    for (auto const &[slot, output] : invocation.outputs) {
      if (output.tensor_guid == tensor_guid) {
        return std::pair{invocation, output};
      }
    }
  }
  return std::nullopt;
}

std::pair<DynamicOpenDataflowGraph, DynamicValueAttrs>
    perform_loss_insertion(DynamicOpenDataflowGraph const &dg,
                           LossAttrs const &loss_attrs,
                           dynamic_tensor_guid_t logit_tensor) {
  auto [logic_invocation, logit_value] =
      assert_unwrap(find_output_tensor(dg, logit_tensor));

  DynamicValueAttrs logit_grad_value{
      // FIXME (Elliott): should we make a new one?
      /*tensor_guid=*/logit_value.tensor_guid,
      /*parallel_tensor_shape=*/logit_value.parallel_tensor_shape,
      /*shard_coord=*/logit_value.shard_coord,
      /*accessor=*/std::nullopt,
      /*role=*/mk_dynamic_tensor_role_loss(),
  };
  DynamicNodeInvocation loss_invocation{
      /*inputs=*/{
          {DynamicTensorSlot{/*slot_name=*/TensorSlotName::LOGIT,
                             /*slot_tensor_role=*/logit_value.role},
           logit_value},
      },
      /*node_attrs=*/
      DynamicNodeAttrs{
          /*task_type=*/DynamicTaskType::LOSS,
          /*device_coord=*/std::nullopt,
          /*mapping=*/std::nullopt,
          /*op_attrs=*/std::nullopt,
          // FIXME (Elliott): should we make a new one?
          /*layer_guid=*/logic_invocation.node_attrs.layer_guid,
          /*per_device_op_state=*/std::nullopt,
      },
      /*outputs=*/
      {
          {DynamicTensorSlot{/*slot_name=*/TensorSlotName::OUTPUT,
                             /*slot_tensor_role=*/logit_grad_value.role},
           logit_grad_value},
      },
  };
  DynamicOpenDataflowGraph result = dg;
  result.invocations.insert(loss_invocation);
  return std::pair{result, logit_grad_value};
}

} // namespace FlexFlow
