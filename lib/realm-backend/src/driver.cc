#include "realm-backend/driver.h"

using namespace Realm;
using namespace FlexFlow;

Logger log_app("app");

int main(int argc, const char **argv) {
  Runtime rt;
  rt.init(&argc, (char ***)&argv);

  Processor::register_task_by_kind(Processor::LOC_PROC, false /*!global*/, 
                                    static_cast<Processor::TaskFuncID>(task_id_t::TOP_LEVEL_TASK_ID),
                                   CodeDescriptor(top_level_task),
                                   ProfilingRequestSet())
      .external_wait();

  Processor p = Machine::ProcessorQuery(Machine::get_machine())
                    .only_kind(Processor::LOC_PROC)
                    .first();
  assert(p.exists());

  rt.shutdown(rt.collective_spawn(p, static_cast<Processor::TaskFuncID>(task_id_t::TOP_LEVEL_TASK_ID), 0, 0));
  return rt.wait_for_shutdown();
}
