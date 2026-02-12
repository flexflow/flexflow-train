#include "task-spec/dynamic_graph/loss_insertion.h"
#include "pcg/optimizer_attrs.h"
#include "task-spec/dynamic_graph/dynamic_layer_guid_t.h"
#include "task-spec/dynamic_graph/dynamic_node_invocation.dtg.h"
#include "task-spec/dynamic_graph/dynamic_open_dataflow_graph.h"
#include "task-spec/dynamic_graph/dynamic_tensor_guid_t.h"
#include "task-spec/dynamic_graph/dynamic_tensor_role.h"
#include "task-spec/dynamic_graph/dynamic_value_attrs.dtg.h"
#include "utils/optional.h"
#include <optional>

namespace FlexFlow {

LossInsertionResult perform_loss_insertion(DynamicOpenDataflowGraph const &dg,
                                           LossAttrs const &loss_attrs,
                                           dynamic_tensor_guid_t logit_tensor) {
  DynamicValueAttrs logit_value = assert_unwrap(
      find_output_value_attrs(dg, logit_tensor, mk_dynamic_tensor_role_fwd()));

  DynamicValueAttrs label_value{
      /*tensor_guid=*/mk_dynamic_tensor_guid_for_loss(),
      /*parallel_tensor_shape=*/logit_value.parallel_tensor_shape,
      /*shard_coord=*/logit_value.shard_coord,
      /*accessor=*/std::nullopt,
      /*role=*/mk_dynamic_tensor_role_loss(),
  };
  DynamicValueAttrs logit_grad_value{
      /*tensor_guid=*/logit_value.tensor_guid,
      /*parallel_tensor_shape=*/logit_value.parallel_tensor_shape,
      /*shard_coord=*/logit_value.shard_coord,
      /*accessor=*/std::nullopt,
      /*role=*/mk_dynamic_tensor_role_bwd(),
  };
  DynamicNodeInvocation loss_invocation{
      /*inputs=*/{
          {DynamicTensorSlot{/*slot_name=*/TensorSlotName::INPUT,
                             /*slot_tensor_role=*/label_value.role},
           label_value},
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
          /*loss_attrs=*/loss_attrs,
          /*layer_guid=*/mk_dynamic_layer_guid_for_loss(),
          /*per_device_op_state=*/std::nullopt,
      },
      /*outputs=*/
      {
          {DynamicTensorSlot{/*slot_name=*/TensorSlotName::LOGIT,
                             /*slot_tensor_role=*/logit_grad_value.role},
           logit_grad_value},
      },
  };
  DynamicOpenDataflowGraph result = dg;
  result.invocations.insert(loss_invocation);
  return LossInsertionResult{result, label_value, logit_grad_value};
}

} // namespace FlexFlow
