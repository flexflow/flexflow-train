The Realm backend for distributed execution.

This is a single-controller implementation. That means the controller (the task that launches all other work) runs on a single node and remotely launches work onto other nodes. Aside from caveats mentioned below, this implementation is (mostly) capable of distributed execution.

Major components:

* `PCGInstance`: the main public interface for the Realm backend. It takes a mapped PCG and lowers it through the dynamic graph to get the fully-specified execution order of tasks to be executed. Besides the usual dynamic graph passes (pass expansion, update insertion, shard expansion), this class also tracks the allocation of Realm instances for tensors.
* `RealmManager`: manages the initialization and shutdown of the Realm runtime. Provides the interface to launch the controller that runs the rest of the computation.
* `RealmContext`: an interface that wraps the rest of Realm and protects against certain classes of bugs, such as shutdown bugs. **Do NOT call Realm directly unless you know what you are doing.**
* `tasks/`: the Realm task implementations and their supporting infrastructure.
  * `impl/`: the actual bodies of Realm tasks, along with interfaces to call them, and the serialization infrastructure for their arguments.
  * `serializer/`: additional support for serializing Realm data types.
  * `realm_task_registry.h`: manages the registration of Realm tasks. All Realm tasks go through this interface.
  * `task_id_t.h` and `realm_task_id_t.h`: types to represent Realm tasks, along with an encoding to Realm's native task ID type.

Other components used mainly within `PCGInstance`:

 * `DistributedDeviceHandle`: represents a distributed device handle (i.e., device handles on all the GPUs on the system), for convenience.
 * `DependenceSet`: tracks dependencies during execution of tasks.
 * `distributed_device_state_initialization.h`: performs device state initialization of dynamic graph nodes and returns the resulting `PerDeviceOpStateBacking`.
 * `instance_allocation.h`: allocates instances for tensors in the dynamic graph and returns the resulting `TensorInstanceBacking`.

TODO list:

* external instances
* copies
* task fusion
* parallel operator implementation (partition, reduce, gather, etc.)
* and fused parallel operators (reduce + broadcast = allreduce)
* memory-optimizing compiler integration (tensor creation/destruction, tensor reuse)
* control replication
* Realm subgraphs
