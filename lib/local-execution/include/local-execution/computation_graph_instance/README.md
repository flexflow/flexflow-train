The primary external-facing interface of local-execution.

Flow: ComputationGraph => ComputationGraphInstance => InitializedComputationGraphInstance

* ComputationGraphInstance takes a ComputationGraph and expands it to generate a DynamicOpenDataflowGraph.
* InitializedComputationGraphInstance takes a ComputationGraph runs initialization to make it ready to run. Once initialized you can execute passes over the graph.
* Tensors and per-device op states are stored directly on the DynamicOpenDataflowGraph.
