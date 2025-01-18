
#include "compiler/cost_estimator/task_graph.dtg.h"
#include "compiler/cost_estimator/tasks_state_tracker.dtg.h"
#include "compiler/cost_estimator/timed_task.dtg.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph.h"
#include "utils/containers/filtrans.h"
#include "utils/containers/is_subseteq_of.h"
#include "utils/exception.h"
#include "utils/graph/digraph/algorithms.h"
#include "utils/graph/digraph/algorithms/get_predecessors.h"
#include "utils/graph/digraph/algorithms/get_successors.h"
#include "utils/overload.h"

namespace FlexFlow {

static void start_task_processing(TasksStateTracker &state_tracker,
                                  TaskGraph const &task_graph,
                                  Node const &task) {
  float cost = task_graph.cost_map.at(task);
  state_tracker.tasks_processing.push(
      TimedTask{task, state_tracker.current_time + cost});
  state_tracker.ready_tasks.erase(task);
}

static bool dependencies_are_satisfied(TasksStateTracker const &state_tracker,
                                       TaskGraph const &task_graph,
                                       Node const &task) {
  std::unordered_set<Node> incoming_dependencies =
      get_predecessors(task_graph.graph, task);
  return is_subseteq_of(incoming_dependencies, state_tracker.processed_tasks);
}

static void finish_task_processing(TasksStateTracker &state_tracker,
                                   TaskGraph const &task_graph,
                                   TimedTask const &timed_task) {
  state_tracker.processed_tasks.insert(timed_task.node);
  for (Node const &task : get_successors(task_graph.graph, timed_task.node)) {
    if (dependencies_are_satisfied(state_tracker, task_graph, task)) {
      state_tracker.ready_tasks.insert(task);
    }
  }
  state_tracker.current_time = timed_task.endtime;
}

static bool is_processing_done(TasksStateTracker const &state_tracker) {
  return state_tracker.ready_tasks.empty() &&
         state_tracker.tasks_processing.empty();
}

static TimedTask get_next_task(TasksStateTracker &state_tracker) {
  TimedTask task = state_tracker.tasks_processing.top();
  state_tracker.tasks_processing.pop();
  return task;
}

float simulate_forward_pass(TaskGraph const &task_graph) {
  TasksStateTracker state_tracker =
      TasksStateTracker{get_sources(task_graph.graph), {}, {}, 0.0};
  while (!is_processing_done(state_tracker)) {
    auto ready_tasks_copy = state_tracker.ready_tasks;
    for (Node const &task : ready_tasks_copy) {
      if (task_graph.is_allowed_to_run(task, state_tracker)) {
        start_task_processing(state_tracker, task_graph, task);
      }
    }

    TimedTask next_task = get_next_task(state_tracker);
    finish_task_processing(state_tracker, task_graph, next_task);
  }

  return state_tracker.current_time;
}
} // namespace FlexFlow
