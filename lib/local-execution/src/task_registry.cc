#include "local-execution/task_registry.h"
#include "local-execution/task_signature_impl.h"
#include "pcg/computation_graph.h"

namespace FlexFlow {

TaskRegistry construct_task_registry(
    std::unordered_map<layer_guid_t, LayerAttrs> const &layer_attrs_mapping) {
  std::unordered_map<layer_guid_t, std::optional<task_id_t>> init_task_ids;
  std::unordered_map<layer_guid_t, std::optional<task_id_t>> fwd_task_ids;
  std::unordered_map<layer_guid_t, std::optional<task_id_t>> bwd_task_ids;

  std::unordered_map<task_id_t, TaskSignatureAndImpl> task_mapping;

  for (std::pair<layer_guid_t, LayerAttrs> const &layer_attrs :
       layer_attrs_mapping) {
    layer_guid_t node = layer_attrs.first;
    init_task_ids.insert({node, std::nullopt});
    fwd_task_ids.insert({node, std::nullopt});
    bwd_task_ids.insert({node, std::nullopt});

    ComputationGraphOpAttrs attrs = layer_attrs.second.op_attrs;
    std::vector<task_id_t> task_ids = get_task_ids(attrs);

    for (task_id_t const &task_id : task_ids) {
      TaskSignatureAndImpl task_signature_impl = get_task_sig_impl(task_id);
      switch (task_signature_impl.task_signature.type) {
        case OpTaskType::INIT:
          assert(is_invocation_valid(task_signature_impl.task_signature,
                                     init(attrs)));
          init_task_ids[node] = task_id;
          break;
        case OpTaskType::FWD:
          assert(is_invocation_valid(task_signature_impl.task_signature,
                                     forward(attrs)));
          fwd_task_ids[node] = task_id;
          break;
        case OpTaskType::BWD:
          assert(is_invocation_valid(task_signature_impl.task_signature,
                                     backward(attrs)));
          bwd_task_ids[node] = task_id;
          break;
        default:
          throw mk_runtime_error(
              fmt::format("Invalid OpTaskType, got {}",
                          task_signature_impl.task_signature.type));
      }
      task_mapping.insert({task_id, task_signature_impl});
    }
  }

  return TaskRegistry{init_task_ids, fwd_task_ids, bwd_task_ids, task_mapping};
}

bool registry_contains_task_for_layer(TaskRegistry const &task_registry,
                                      layer_guid_t const &op,
                                      OpTaskType const &op_task_type) {
  std::unordered_map<layer_guid_t, std::optional<task_id_t>> task_ids;
  switch (op_task_type) {
    case OpTaskType::INIT:
      task_ids = task_registry.init_task_ids;
      break;
    case OpTaskType::FWD:
      task_ids = task_registry.forward_task_ids;
      break;
    case OpTaskType::BWD:
      task_ids = task_registry.backward_task_ids;
      break;
    default:
      throw mk_runtime_error(
          fmt::format("Invalid OpTaskType, got {}", op_task_type));
  }

  assert(task_ids.count(op));
  return task_ids.at(op).has_value();
}

} // namespace FlexFlow
