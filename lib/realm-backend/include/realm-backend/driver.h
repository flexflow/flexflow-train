#ifndef _FLEXFLOW_REALM_BACKEND_DRIVER_H
#define _FLEXFLOW_REALM_BACKEND_DRIVER_H

#include "realm.h"
#include "realm/cmdline.h"
#include "task-spec/op_task_invocation.h"

void top_level_task(const void *args, size_t arglen, const void *userdata,
               size_t userlen, Realm::Processor p);

#endif
