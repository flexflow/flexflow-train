# graph

## Design Considerations

FlexFlow's graph library very intentionally attempts to balance performance and ease of use. 
The graph library aims to have a very simple external interface that is highly decoupled from the underlying representations, so performance and internal implementations can be tuned and modified over time without breaking the code that uses the library.
Because FlexFlow's graphs are not on the scale of machine memory or not so large that single traversals takes nontrivial time, the graph library intentionally avoids performance opportunities that would expose many of these performance aspects to user code.
Of course, there are also some optimizations that simply have not been done due to time constraints: for example, algorithms currently are able to be specialized for the underlying representation being used, but this could be added without modifying the user-side interface.

## Usage

### Core Graph Variants

There is no single type of graph. Should it be directed? Allow multiple edges between nodes? Should nodes and/or edges have information attached?
Because there is no single answer to this question, similar to [networkx](https://networkx.org/) we provide a number of different graph variants. 
At their core, they are as follows:

- `UndirectedGraph`: at most one edge allowed between every pair of nodes, edges are undirected.
- `DiGraph`: at most one edge allowed between every ordered pair of nodes, edges are directed (i.e., have a source node and a destination node)
- `MultiDiGraph`: arbitrary numbers of directed edges allowed between every pair of nodes.
- `DataflowGraph`: used to model computation graphs. See the [DataflowGraph](#dataflowgraph) section for a detailed explanation.

Examples of the different graph variants are shown below.

Example of `UndirectedGraph`:
```mermaid
flowchart TD
    A(" ")
    B(" ")
    C(" ")
    D(" ")
    E(" ")
    
    A --- B
    A --- C
    B --- C
    B --- B
    D --- B
```

Example of `DiGraph`:
```mermaid
flowchart TD
    A(" ")
    B(" ")
    C(" ")
    D(" ")
    E(" ")
    F(" ")

    A --> F
    B --> E
    B --> C
    B --> B
    D --> B
    C --> D
```

Example of `MultiDiGraph`:
```mermaid
flowchart TD
    A
    B
    C
    D
    E
    F

    A --> B
    B --> C
    C --> D
    D --> A
    B --> E
    E --> B
    D --> A
    A --> E
    D --> D
    E --> E
```

Note that the node names are completely arbitrary: they have no apparent ordering or other meaning besides representing the topology of the graph.
This is the case with all of the 4 core graph classes.
Nodes are of type `Node`, and from a user perspective are simply opaque handles, and source and destination indices should similarly be considered opaque from a user point of view.
In addition, nodes should only be used in the context of their graph, so comparing or checking equality of nodes between different graphs (even of the same type) is undefined behavior[^1].

All three core graph variants allow insertion and deletion of both edges and nodes. 
To add a node to an `UndirectedGraph g`, simply call `g.add_node()`, which will return a `Node` object.
For semantics closer to `networkx`'s method of adding nodes, `g.add_node_unsafe(my_node)` can be used. This is useful when constructing a modified copy of an existing graph (given that it maintains node bijection), though it is not generally recommended. 
The interface for node addition is identical for `DiGraph` and `MultiDiGraph`.
To add an edge between two nodes `Node n1` and `Node n2` to an `UndirectedGraph g`, call `g.add_edge({n1, n2})`.
In `UndirectedGraph` the order of the arguments of `add_edge` doesn't matter as edges are undirected, but the order does matter for `DiGraph`, `MultiDiGraph` and `DataflowGraph`.

The last paragraph covered the base API used to write to graphs, but we also want to be able to read from graphs.
Reading from graphs is implemented with the `query_nodes` and `query_edges` methods, which can be thought of as executing a database query over the nodes and edges of the target graph, respectively (where queries are restricted to an incredibly simple set of operations).
The argument to `query_nodes` is a `NodeQuery` (which is simply a set of `Node`s).
`query_nodes` then returns the intersection of the nodes in the graph and the nodes in the query. 
The set of nodes in the query is actually an `optional`, so `nullopt` could also be passed, which would simply retrieve all nodes from the target graph (essentially `nullopt` acts as the set of all nodes that could ever exist).
`query_edges` functions similarly, but as with `add_edge` its behavior is differs slightly between the three graph variants.
`UndirectedGraph::query_edges` simply takes an optional set of nodes and returns all edges that touch any of those nodes.
`DiGraph::query_edges` allows separate sets for source and destination nodes, and `MultiDiGraph::query_edges` adds the ability to filter by source and destination indices as well.

In practice you will rarely ever use `query_nodes` and `query_edges` as the graph library provides a large number of algorithms that do that work for you, but it can be helpful to understand this base layer if you ever need to implement your own algorithms.
The layer users will most commonly interact with is the interface provided within either the `algorithms.h` header files or the `algorithms` folders, present in their respective graph class folders.
They provide a large number of pre-implemented algorithms on graphs, ranging from as simple as `get_nodes` to as complex as `get_transitive_reduction` and `get_dominators`.
Note that, due to the internal virtual inheritance structure, some functions for more privitive classes can be employed by the derived classes. (For example, `get_nodes` present in `node/algorithms.h` can be used by `DiGraph`).
You may notice that the most of algorithms present take as arguments not `UndirectedGraph`, `DiGraph`, and `MultiDiGraph`, but rather `UndirectedGraphView`, `DiGraphView`, and `MultiDiGraphView`. 
These `GraphView` objects represent read-only (i.e., immutable) graphs.
Similar to C++'s `const` semantics, `Graph`s can be coerced[^2] to `GraphView`s but not the other way around.
To transform a `GraphView` to a `Graph`, we can perform an explicit copy with `materialize_view`.
Both `Graph` and `GraphView` types follow normal value semantics. 
This may seem wasteful (oftentimes graphs are large objects that are passed around via reference to avoid making additional copies), but the `Graph` and `GraphView` types internally implement copy-on-write optimizations to only perform the minimum number of actual copies while maintaining immutability and lifetime safety (if you allocate a `DiGraph` use for example `get_subgraph` to get a `DiGraphView` representing a part of this graph, modifications to the underlying `DiGraph` will not be mirrored in the `DiGraphView` and the `DiGraphView` will remain valid even after the base `DiGraph` leaves scope.

At this point, however, we still have not discussed how to create a graph.
The user-facing graph interface is intentionally separated from the underlying graph representations, so representations can be changed without requiring any user-side code modifications besides the choice of which implementation to use.
For example, to construct a `DiGraph` which internally uses a representation such as `AdjacencyDiGraph` we do the following:
```cpp
DiGraph g = DiGraph::create<AdjacencyDiGraph>();
```
Generally users will use underlying representations provided by the graph library, but advanced users can create their own implementations (see the [Internals](#internals) section).

[^1]: At some point we will likely add actual runtime checks on this, but for now we rely on the user not to mess up. Currently the implementation will keep going silently until the incorrectness grows so large that something breaks/crashes.
[^2]: See <https://en.wikipedia.org/wiki/Type_conversion> if you're not familiar with the term _type coercion_

### DataflowGraph

The primary abstraction for representing computation graphs / task graphs is the `DataflowGraph` interface (along with its variants, `OpenDataflowGraph`, `LabelleledDataflowGraph` and `OpenLabelleledDataflowGraph`).
At a high level, nodes represent multivariate functions (from tuples of inputs to tuple of outputs), while edges represent value uses of such functions.

`DataflowGraph` is similar to `MultiDiGraph`, but with the following important differences:
  - The edges entering, exiting a given nodes have a well-defined order.
  - The outputs of a given node also have a well-defined order. 
  - `DataflowGraph`s are directed acyclic graphs. This is enforced by the interface used to construct them, since a node can only be added to the graph after all of its predecessor nodes have already been added.

The main components of `DataflowGraph` are as follows:
- `DataflowInput`: used to denote an entry in the ordered sequence of incoming dependencies (arguments) of a given node (operator). 
- `DataflowOutput`: used to denote an entry in the ordered sequence of outgoing results (value uses) from a given node (operator).
- `DataflowEdge`: wrapper around a `DataflowInput`, `DataflowOutput` pair between 2 nodes.
- `NodeAddedResult`: returned upon adding a new node. Contains the newly generated `Node` and the vector of `DataflowOutput`s for the given node.

`DataflowGraph`s are constructed as follows:

```cpp
    auto g = DataflowGraph::create<UnorderedSetDataflowGraph>();
    
    // Node with no inputs and 2 outputs
    NodeAddedResult n1_result = g.add_node({}, 2);
    Node n1 = n1_result.node;
    DataflowOutput n1_o1 = n1_result.outputs[0];
    DataflowOutput n1_o2 = n1_result.outputs[1];

    // Node with 2 inputs and 1 output
    NodeAddedResult n2_result = g.add_node({n1_o1, n1_o2}, 1);
    Node n2 = n2_result.node;
    DataflowOutput n2_o1 = n2_result.outputs[0];

    // Node with 1 input and 2 outputs
    NodeAddedResult n3_result = g.add_node({n1_o2}, 1);
    Node n3 = n3_result.node;
    DataflowOutput n3_o1 = n3_result.outputs[0];
    DataflowOutput n3_o2 = n3_result.outputs[1];

    // Node with 2 inputs and 1 output
    NodeAddedResult n4_result = g.add_node({n2_o1, n3_o1}, 1);
    Node n4 = n4_result.node;
    DataflowOutput n4_o1 = n4_result.outputs[0];
```

which generates the following graph

```mermaid
flowchart TD
    subgraph Node1[ ]
        direction TB
        N1Process[n1]
        n1_o1((n1_o1))
        n1_o2((n1_o2))
        N1Process --> n1_o1
        N1Process --> n1_o2
    end

    subgraph Node2[ ]
        direction TB
        n2_i1((n2_i1))
        n2_i2((n2_i2))
        N2Process[n2]
        n2_o1((o1))
        n2_i1 --> N2Process
        n2_i2 --> N2Process
        N2Process --> n2_o1
    end

    subgraph Node3[ ]
        direction TB
        n3_i1((n3_i1))
        N3Process[n3]
        n3_o1((n3_o1))
        n3_o2((n3_o2))
        n3_i1 --> N3Process
        N3Process --> n3_o1
        N3Process --> n3_o2
    end

    subgraph Node4[ ]
        direction TB
        n4_i1((n4_i1))
        n4_i2((n4_i2))
        N4Process[n4]
        n4_o1((n4_o1))
        n4_i1 --> N4Process
        n4_i2 --> N4Process
        N4Process --> n4_o1
    end

    n1_o1 --> n2_i1
    n1_o2 --> n2_i2
    n1_o2 --> n3_i1
    n2_o1 --> n4_i1
    n3_o1 --> n4_i2
```


### Open Dataflow Variant

`Open` should be interpreted in the topological sense: that is, a graph that contains some edges where one of the edge's 2 nodes is not present in the graph itself.
This graph class is particularly useful for processing a sub-graph of a given graph while still maintaining information regarding the edges that cross the cut.
`DataflowGraphInput` is used to represent the open (incoming) inputs to the graph. Note that, unlike `DataFlowInput`, `DataflowGraphInput`s are unordered (given that they are inputs to possibly several different nodes within the graph).

### Labelled Dataflow Variant

As nice as all of the above is, graphs without labels are mostly useless--in practice, nodes and edges represent some other system and the properties of that system (or at least a way to map the result of graph algorithms back to the underlying system) are necessary.
Thus, FlexFlow's graph library provides the ability to add labels to `DataflowGraph`, through the `LabelleledDataflowGraph` and `OpenLabelleledDataflowGraph`, which allow users to label different components of the graph. 
- `LabelledDataflowGraph` allows for labelling of `Node`s and `DataflowOutput`s.
- `OpenLabelledDataflowGraph` allows for labelling of `Node`s and `OpenDataflowValue`s, which is a variant describing both `DataflowOutput`s and `DataflowGraphInput`s.

While the interfaces of these graphs differ slightly from the core graph variants, they still have the corresponding `add_node` methods, and `query_nodes`/`query_edges` methods. (Note that there is no `add_edge` method since, for `DataflowGraph`, edges are implicitly added when we add a node and specify its predecessors)
Note that all of the labelled graph types require that each element of the labelled types have a label, which is enforced via the interfaces they provide.
Partial labelling can be implement via wrapping the label type in `optional`.
Interacting with `Node` and `Edge` objects is still necessary to use the labelled graph types: intuitively the labelled graph types can be thought of as a pair of a core graph variant and a hash map the maps nodes/edges to labels.
As such, the labelled graph types provide the typical `at` method (as on `std::unordered_map`[^3]) and can be coerced to their underlying core graph variants.

[^3]: `operator[]` currently is not present because all nodes must have labels and we don't require label types to be default constructible, though some simple template programming could probably add `operator[]` support in the cases where the label types _are_ default constructible.


## Internals

Most of the major graph classes in the library come in sets of 4. For a given class `GlassName` we have:
1. `ClassName`
2. `ClassNameView`
3. `IClassName`
4. `IClassNameView`

General rules which apply to most classes:
- `ClassName` (virtually) inherits from `ClassNameView`. Similarly, `IClassName` (virtually) inherits from `IClassNameView`.
- `ClassName` has, as a member variable, a `cow_ptr` of type `IClassName`. Same holds for `ClassNameView`.
Thus, the bulk of the inheritance that actually extends functionality is present among `IClassNameView` classes. 


### cow_ptr and Interfaces

The reason for the existence of the `View` variants has been explained in previous sections.
The existence of the `I(nterface)` variants stems from C++'s approach to modeling polymorphism.

C++ polymorphism is achieved at runtime through the use of [virtual functions](https://www.learncpp.com/cpp-tutorial/virtual-functions/), which allow for a single function defined on some superclass to also work correctly on its subclasses.

To create objects with polymorphic behaviour, we use the following syntax:
`BaseClass* obj = new DerivedClass(); //or alternatives such as std::shared_ptr<BaseClass> obj = std::make_shared<DerivedClass>();`
Any call to `obj`'s member functions are resolved at runtime (dynamic binding), with C++ calling the most derived implementation of the function.

While this pattern works nicely, the way instantiation is done leaves the burden of memory management on the user.
To address this, graph classes store a `cow_ptr` as a member variable, which point to instances of type equal to their corresponding interface class.

All member functions present in `ClassName` and `ClassNameView` delegate their calls to their corresponding interface classes (which implement the actual logic), meaning that these classes essentially act as wrappers to their interface counterparts.

### Virtual Inheritance
Due to the complexity of the graph library, diamond-style inheritance patterns emerge.
In the case of a diamond inheritance pattern, C++ will instantiate multiple copies of the base class whenever we instantiate a derived class.
To address this issue, we employ [Virtual Inheritance](https://en.wikipedia.org/wiki/Virtual_inheritance), which removes the ambiguity associated with the multiple copies.
