#ifndef _FLEXFLOW_REALM_BACKEND_TASK_WRAPPER_H
#define _FLEXFLOW_REALM_BACKEND_TASK_WRAPPER_H

#include "local-execution/local_task_registry.h"
#include "realm-backend/task_result.h"

namespace FlexFlow {

/* The following are general task wrappers to be invoked by the Realm runtime */

template <typename T> struct RealmTaskArgs {
  task_id_t task_id;
  TaskImplFunction impl_function;
  TaskArgumentAccessor accessor;
  Promise<T> promise;
};

void init_wrapper_task(const void *args, size_t arglen, const void *userdata,
                       size_t userlen, Realm::Processor p);

void fwdbwd_wrapper_task(const void *args, size_t arglen, const void *userdata,
                         size_t userlen, Realm::Processor p);

void generic_wrapper_task(const void *args, size_t arglen, const void *userdata,
                          size_t userlen, Realm::Processor p);

void register_wrapper_tasks_init(int p_id, Realm::Processor p, task_id_t task_id);

void register_wrapper_tasks_fwdbwd(int p_id, Realm::Processor p, task_id_t task_id);

void register_wrapper_tasks_generic(int p_id, Realm::Processor p, task_id_t task_id);

void register_wrapper_tasks(int pid, Realm::Processor p, task_id_t task_id,
                            TaskSignatureAndImpl task_sig_impl);

} // namespace FlexFlow

#endif