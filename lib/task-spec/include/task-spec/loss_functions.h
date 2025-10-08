/* Copyright 2023 CMU, Facebook, LANL, MIT, NVIDIA, and Stanford (alphabetical)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_LOSS_FUNCTIONS_H
#define _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_LOSS_FUNCTIONS_H

#include "op-attrs/ops/loss_functions.h"
#include "task-spec/symbolic_forward_tensor_guid_t.dtg.h"
#include "task-spec/symbolic_gradient_tensor_guid_t.dtg.h"
#include "task-spec/symbolic_loss_tensor_guid_t.dtg.h"
#include "task-spec/task_impl_function.dtg.h"
#include "task-spec/runtime_task_invocation.dtg.h"
#include "task-spec/runtime_task_signature.h"

namespace FlexFlow {

TaskImplFunction get_loss_bwd_task_impl();
RuntimeTaskSignature get_loss_bwd_signature();
RuntimeTaskInvocation loss_attrs_backward(LossAttrs const &,
                        symbolic_forward_tensor_guid_t logit,
                        symbolic_gradient_tensor_guid_t logit_grad,
                        symbolic_loss_tensor_guid_t label);

} // namespace FlexFlow

#endif
