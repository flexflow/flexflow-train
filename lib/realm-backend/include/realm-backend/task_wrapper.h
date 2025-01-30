#ifndef _FLEXFLOW_REALM_BACKEND_TASK_WRAPPER_H
#define _FLEXFLOW_REALM_BACKEND_TASK_WRAPPER_H

#include "local-execution/task_registry.h"
#include "realm-backend/driver.h"
#include "realm-backend/realm_task_argument_accessor.h"

namespace FlexFlow {

/* The following are general task wrappers to be invoked by the Realm runtime */

struct RealmTaskArgs {
  task_id_t task_id;
  TaskImplFunction impl_function;
  TaskArgumentAccessor accessor;
  void *result;
};

void init_wrapper_task(const void *args, size_t arglen, const void *userdata,
                       size_t userlen, Realm::Processor p);

void fwdbwd_wrapper_task(const void *args, size_t arglen, const void *userdata,
                         size_t userlen, Realm::Processor p);

void generic_wrapper_task(const void *args, size_t arglen, const void *userdata,
                          size_t userlen, Realm::Processor p);

void register_wrapper_tasks_init(Realm::Processor p, task_id_t task_id);

void register_wrapper_tasks_fwdbwd(Realm::Processor p, task_id_t task_id);

void register_wrapper_tasks_generic(Realm::Processor p, task_id_t task_id);

void register_wrapper_tasks(Realm::Processor p, task_id_t task_id,
                            TaskSignatureAndImpl task_sig_impl);

} // namespace FlexFlow

#endif