The primary external-facing interface of local-execution.

Flow:

* input (from compiler): `ComputationGraph`
* `create_computation_graph_instance()` => `ComputationGraphInstance`
* `initialize_computation_graph_instance()` => `InitializedComputationGraphInstance`
* execute (TBD)

Details:

* `ComputationGraph` is the unexpanded form of the graph: no passes, no parallelism, etc.
* `create_computation_graph_instance()` takes the `ComputationGraph` and expands it into a `DynamicOpenDataflowGraph`. This form has passes and updates but no allocations and no parallelism. (Note because this is the *local* executor there will be no parallelism.) This version gets stored in the `ComputationGraphInstance`.
* `initialize_computation_graph_instance()` takes the `ComputationGraphInstance`, along with user-provided input tensors. It allocates any remaining (not-user-provided) tensors and performs initialization (cuBLAS handles, etc.). These get stored in a new `DynamicOpenDataflowGraph` which gets wrapped in `InitializedComputationGraphInstance`. (The old `DynamicOpenDataflowGraph` is treated as immutable and is not modified.) This form is fully specified and ready for (single-device) execution.
