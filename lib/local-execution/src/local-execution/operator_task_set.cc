#include "local-execution/operator_task_set.h"
#include "local-execution/registered_task.h"
#include "task-spec/task_signature_impl.h"
#include "utils/bidict/algorithms/right_entries.h"
#include "utils/containers/values.h"

namespace FlexFlow {

bidict<OpTaskType, registered_task_t>
    get_map_from_task_type_to_task(OperatorTaskSet const &op_task_set) {
  return {
      {OpTaskType::INIT, op_task_set.init_task},
      {OpTaskType::FWD, op_task_set.fwd_task},
      {OpTaskType::BWD, op_task_set.bwd_task},
  };
}

std::unordered_set<registered_task_t>
    get_all_tasks_in_task_set(OperatorTaskSet const &op_task_set) {
  return right_entries(get_map_from_task_type_to_task(op_task_set));
}

registered_task_t get_task_for_task_type(OperatorTaskSet const &op_task_set,
                                         OpTaskType task_type) {
  return get_map_from_task_type_to_task(op_task_set).at_l(task_type);
}

OperatorTaskSet
    get_task_set_for_operator(ComputationGraphOpAttrs const &attrs) {
  registered_task_t init_task = make_noop_registered_task();
  registered_task_t fwd_task = make_noop_registered_task();
  registered_task_t bwd_task = make_noop_registered_task();

  std::vector<task_id_t> task_ids = get_task_ids(attrs);

  for (task_id_t const &task_id : task_ids) {
    TaskSignatureAndImpl task_signature_and_impl =
        get_task_signature_and_impl_for_task_id(task_id);

    TaskImplFunction task_impl_function = task_signature_and_impl.impl_function;
    OpTaskSignature task_signature = task_signature_and_impl.task_signature;

    switch (task_signature.type) {
      case OpTaskType::INIT:
        ASSERT(is_invocation_valid(task_signature,
                                   get_init_op_task_invocation(attrs)));
        init_task = registered_task_t{task_id};
        break;
      case OpTaskType::FWD:
        ASSERT(is_invocation_valid(task_signature,
                                   get_forward_op_task_invocation(attrs)));
        fwd_task = registered_task_t{task_id};
        break;
      case OpTaskType::BWD:
        ASSERT(is_invocation_valid(task_signature,
                                   get_backward_op_task_invocation(attrs)));
        bwd_task = registered_task_t{task_id};
        break;
      default:
        PANIC("Unhandled OpTaskType", fmt::to_string(task_signature.type));
    }
  }

  return OperatorTaskSet{
      /*init_task=*/init_task,
      /*fwd_task=*/fwd_task,
      /*bwd_task=*/bwd_task,
  };
}

} // namespace FlexFlow
