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

#ifndef _FF_METRICS_FUNCTIONS_H_
#define _FF_METRICS_FUNCTIONS_H_

#include "kernels/metric.h"
#include "kernels/perf_metrics.h"
#include "legion.h"
#include "task_spec/task_invocation.h"

namespace FlexFlow {

TypedIndexTaskInvocation<PerfMetrics>
    compute_metrics(MetricsAttrs const &,
                    parallel_tensor_guid_t const &logit,
                    parallel_tensor_guid_t const &label);
TypedStandardTaskInvocation<PerfMetrics>
    update_metrics(MetricsAttrs const &,
                   StandardTypedTaskArg<PerfMetrics> const &all_metrics,
                   IndexTypedTaskArg<PerfMetrics> const &one_metrics);
TypedStandardTaskInvocation<PerfMetrics> compute_and_update_metrics(
    MetricsAttrs const &metrics,
    StandardTypedTaskArg<PerfMetrics> const &all_metrics,
    parallel_tensor_guid_t const &logit,
    parallel_tensor_guid_t const &label);
TaskInvocation reset_metrics(MetricsAttrs const &);

template <>
void register_task<METRICS_COMP_TASK_ID>();
template <>
void register_task<UPDATE_METRICS_TASK_ID>();

} // namespace FlexFlow

VISITABLE_STRUCT(::FlexFlow::MetricsAttrs,
                 loss_type,
                 measure_accuracy,
                 measure_categorical_crossentropy,
                 measure_sparse_categorical_crossentropy,
                 measure_mean_squared_error,
                 measure_root_mean_squared_error,
                 measure_mean_absolute_error);

#endif
