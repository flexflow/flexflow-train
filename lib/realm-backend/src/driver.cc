#include "realm-backend/driver.h"

using namespace Realm;
using namespace FlexFlow;

Processor::TaskFuncID get_realm_task_id(task_id_t task_id) {
  return static_cast<Processor::TaskFuncID>(task_id) +
         Processor::TASK_ID_FIRST_AVAILABLE;
}

int main(int argc, char **argv) {
  Runtime rt;
  rt.init(&argc, &argv);

  Processor::register_task_by_kind(
      Processor::LOC_PROC, false /*!global*/,
      get_realm_task_id(task_id_t::TOP_LEVEL_TASK_ID),
      CodeDescriptor(top_level_task), ProfilingRequestSet())
      .external_wait();

  Processor p = Machine::ProcessorQuery(Machine::get_machine())
                    .only_kind(Processor::LOC_PROC)
                    .first();
  assert(p.exists());

  Event e = rt.collective_spawn(
      p, get_realm_task_id(task_id_t::TOP_LEVEL_TASK_ID), 0, 0);
  rt.shutdown(e);

  return rt.wait_for_shutdown();
}
