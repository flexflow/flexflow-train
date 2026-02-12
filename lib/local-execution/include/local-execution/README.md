The primary external-facing interface of local-execution.

Major components:

* `computation_graph_instance.h`: is the main external facing interface
  * Takes a `ComputationGraph` as input, expands and initializes it
  * Provides various methods to run all or a subset of passes
* `local_task_registry.h`: functions to retrieve task implementations
  * Not a dynamic registry: tasks are all static now
* `local_task_argument_accessor.h`: local wrapper for `ITaskArgumentAccessor`
  * Stores all of the necessary data required for a task to execute
* `task_execution.h`: utilities to prepare and execute tasks
* `tensor_allocation.h`: a pass for the dataflow graph that allocates all tensors
