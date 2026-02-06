#include "realm-execution/realm.h"
#include "realm-execution/realm_task_id_t.h"
#include "realm-execution/task_id_t.dtg.h"

namespace FlexFlow {

void op_task_wrapper(
    void const *, size_t, void const *, size_t, Realm::Processor) {}

static Realm::Event register_task(Realm::Processor::Kind target_kind,
                                  task_id_t func_id,
                                  void (*task_body)(void const *,
                                                    size_t,
                                                    void const *,
                                                    size_t,
                                                    Realm::Processor)) {
  return Realm::Processor::register_task_by_kind(
      target_kind,
      /*global=*/false,
      get_realm_task_id_for_task_id(func_id),
      Realm::CodeDescriptor(task_body),
      Realm::ProfilingRequestSet());
}

Realm::Event register_all_tasks() {
  std::vector<Realm::Event> pending_registrations;

  std::vector<task_id_t> init_task_ids = {
      task_id_t::BATCHNORM_INIT_TASK_ID,
      task_id_t::COMBINE_INIT_TASK_ID,
      task_id_t::CONV2D_INIT_TASK_ID,
      task_id_t::DROPOUT_INIT_TASK_ID,
      task_id_t::ELEMENTBINARY_INIT_TASK_ID,
      task_id_t::ELEMENTUNARY_INIT_TASK_ID,
      task_id_t::GATHER_INIT_TASK_ID,
      task_id_t::LAYERNORM_INIT_TASK_ID,
      task_id_t::LINEAR_INIT_TASK_ID,
      task_id_t::ATTENTION_INIT_TASK_ID,
      task_id_t::POOL2D_INIT_TASK_ID,
      task_id_t::REDUCE_INIT_TASK_ID,
      task_id_t::REDUCTION_INIT_TASK_ID,
      task_id_t::REPARTITION_INIT_TASK_ID,
      task_id_t::REPLICATE_INIT_TASK_ID,
      task_id_t::SOFTMAX_INIT_TASK_ID,
  };

  for (task_id_t init_task_id : init_task_ids) {
    pending_registrations.push_back(register_task(
        Realm::Processor::LOC_PROC, init_task_id, op_task_wrapper));
  }

  return Realm::Event::merge_events(pending_registrations);
}

} // namespace FlexFlow
