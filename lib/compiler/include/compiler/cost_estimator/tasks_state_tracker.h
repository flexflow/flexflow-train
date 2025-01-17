#ifndef _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_COST_ESTIMATOR_TASKS_STATE_TRACKER_H
#define _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_COST_ESTIMATOR_TASKS_STATE_TRACKER_H

#include "compiler/cost_estimator/tasks_state_tracker.dtg.h"
#include "compiler/cost_estimator/timed_component.dtg.h"
#include "compiler/cost_estimator/timed_dependency.dtg.h"
#include "compiler/cost_estimator/timed_layer.dtg.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph_edge.h"
#include "pcg/parallel_computation_graph/parallel_layer_guid_t.dtg.h"

namespace FlexFlow {

void start_layer_processing(TasksStateTracker &state_tracker,
                            parallel_layer_guid_t const &layer);

void start_dependency_processing(
    TasksStateTracker &state_tracker,
    ParallelComputationGraphEdge const &dependency);

void finish_layer_processing(TasksStateTracker &state_tracker,
                             TimedLayer const &timed_layer);

bool is_layer_ready(TasksStateTracker const &state_tracker,
                    parallel_layer_guid_t const &layer);

void finish_dependency_processing(TasksStateTracker &state_tracker,
                                  TimedDependency const &timed_dependency);

bool is_processing_done(TasksStateTracker const &state_tracker);

TimedComponent
    finish_processing_next_component(TasksStateTracker &state_tracker);

} // namespace FlexFlow

#endif
