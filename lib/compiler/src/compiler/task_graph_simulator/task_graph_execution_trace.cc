#include "compiler/task_graph_simulator/task_graph_execution_trace.h"
#include "utils/containers/maximum.h"
#include "utils/containers/transform.h"
#include "utils/exception.h"
#include "utils/fmt/unordered_set.h"

namespace FlexFlow {

float get_endtime(TaskGraphExecutionTrace const &trace) {
  if (trace.task_profiles.empty()) {
    throw mk_runtime_error(
        fmt::format("TaskGraphExecutionTrace {} is empty", trace));
  }
  return maximum(transform(trace.task_profiles, [](TaskProfile const &profile) {
    return profile.end_time;
  }));
}

} // namespace FlexFlow
