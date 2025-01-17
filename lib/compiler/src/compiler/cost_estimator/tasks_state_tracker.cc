
#include "compiler/cost_estimator/tasks_state_tracker.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph.h"
#include "utils/containers/filtrans.h"
#include "utils/containers/is_subseteq_of.h"
#include "utils/exception.h"
#include "utils/overload.h"

namespace FlexFlow {

void start_layer_processing(TasksStateTracker &state_tracker,
                            parallel_layer_guid_t const &layer) {
  float cost = state_tracker.layer_cost_estimator(layer);
  state_tracker.component_processing.push(
      TimedComponent{TimedLayer{state_tracker.current_time + cost, layer}});
  state_tracker.ready_layers.erase(layer);
}

void start_dependency_processing(
    TasksStateTracker &state_tracker,
    ParallelComputationGraphEdge const &dependency) {
  float cost = state_tracker.dependency_cost_estimator(dependency);
  state_tracker.component_processing.push(TimedComponent{
      TimedDependency{state_tracker.current_time + cost, dependency}});
}

void finish_layer_processing(TasksStateTracker &state_tracker,
                             TimedLayer const &timed_layer) {
  state_tracker.processed_components.insert(TimedComponent{timed_layer});
  state_tracker.current_time = timed_layer.endtime;
  std::unordered_set<ParallelComputationGraphEdge> outgoing_dependencies =
      get_outgoing_edges(state_tracker.pcg, timed_layer.layer);
  for (ParallelComputationGraphEdge const &dep : outgoing_dependencies) {
    start_dependency_processing(state_tracker, dep);
  }
}

bool is_layer_ready(TasksStateTracker const &state_tracker,
                    parallel_layer_guid_t const &layer) {
  std::unordered_set<ParallelComputationGraphEdge> incoming_dependencies =
      get_incoming_edges(state_tracker.pcg, layer);

  std::unordered_set<ParallelComputationGraphEdge>
      non_timed_processed_dependencies = filtrans(
          state_tracker.processed_components,
          [](TimedComponent const &component) {
            return component.visit<std::optional<ParallelComputationGraphEdge>>(
                overload{
                    [&](TimedLayer const &layer) { return std::nullopt; },
                    [&](TimedDependency const &dep) { return dep.raw_edge; }});
          });

  return is_subseteq_of(incoming_dependencies,
                        non_timed_processed_dependencies);
}

void finish_dependency_processing(TasksStateTracker &state_tracker,
                                  TimedDependency const &timed_dependency) {
  state_tracker.processed_components.insert(TimedComponent{timed_dependency});
  parallel_layer_guid_t destination_layer =
      get_dst_layer(timed_dependency.raw_edge);

  if (is_layer_ready(state_tracker, destination_layer)) {
    state_tracker.ready_layers.insert(destination_layer);
  }

  state_tracker.current_time = timed_dependency.endtime;
}

bool is_processing_done(TasksStateTracker const &state_tracker) {
  return state_tracker.ready_layers.empty() &&
         state_tracker.component_processing.empty();
}

TimedComponent process_next_component(TasksStateTracker &state_tracker) {
  if (state_tracker.component_processing.empty()) {
    throw mk_runtime_error(
        "Processing queue is empty, cannot process next component");
  }

  TimedComponent component = state_tracker.component_processing.top();
  state_tracker.component_processing.pop();

  if (component.has<TimedDependency>()) {
    finish_dependency_processing(state_tracker,
                                 component.get<TimedDependency>());
  } else if (component.has<TimedLayer>()) {
    finish_layer_processing(state_tracker, component.get<TimedLayer>());
  }
  return component;
}
} // namespace FlexFlow
