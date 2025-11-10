#include "task-spec/dynamic_graph/dynamic_open_dataflow_graph.h"
#include "utils/containers/flatmap.h"
#include "utils/containers/zip_strict.h"
#include "utils/graph/dataflow_graph/algorithms.h"
#include "utils/graph/instances/unordered_set_labelled_open_dataflow_graph.h"
#include "utils/graph/node/algorithms.h"
#include "utils/graph/open_dataflow_graph/algorithms/get_inputs.h"
#include "utils/many_to_one/many_to_one.h"
#include "utils/containers/all_of.h"

namespace FlexFlow {

DynamicOpenDataflowGraph make_empty_dynamic_open_dataflow_graph() {
  return DynamicOpenDataflowGraph{
    LabelledOpenDataflowGraph<DynamicNodeAttrs, DynamicValueAttrs>::create<
      UnorderedSetLabelledOpenDataflowGraph<DynamicNodeAttrs, DynamicValueAttrs>>(),
    std::nullopt,
  };

}

bool full_dynamic_graph_satisfies(
  DynamicOpenDataflowGraph const &,
  std::function<bool(DynamicNodeAttrs const &)> const &,
  std::function<bool(DynamicValueAttrs const &)> const &) {

  NOT_IMPLEMENTED();
}

bool no_part_of_dynamic_graph_satisfies(
  DynamicOpenDataflowGraph const &,
  std::function<bool(DynamicNodeAttrs const &)> const &,
  std::function<bool(DynamicValueAttrs const &)> const &) {

  NOT_IMPLEMENTED();
}

std::unordered_set<DynamicNodeAttrs> get_dynamic_node_set() {
  NOT_IMPLEMENTED();
}

std::unordered_set<DynamicNodeInvocation> get_dynamic_invocation_set(DynamicOpenDataflowGraph const &g) {
  return transform(
    get_nodes(g.raw),
    [&](Node const &n) -> DynamicNodeInvocation {
      std::vector<OpenDataflowValue> n_inputs = get_inputs(g.raw, n);
      std::vector<DataflowOutput> n_outputs = get_outputs(g.raw, n);

      std::vector<DynamicValueAttrs> inputs = 
        transform(n_inputs, 
                  [&](OpenDataflowValue const &v) -> DynamicValueAttrs {
                    return g.raw.at(v);
                  });

      std::vector<DynamicValueAttrs> outputs = 
        transform(n_outputs,
                  [&](DataflowOutput const &v) -> DynamicValueAttrs {
                    return g.raw.at(OpenDataflowValue{v});
                  });
    
      return DynamicNodeInvocation{
        /*inputs=*/inputs,
        /*node_attrs=*/g.raw.at(n),
        /*outptuts=*/outputs,
      };
    });
}

DynamicOpenDataflowGraph
  transform_dynamic_invocation_set(
    DynamicOpenDataflowGraph const &,
    std::function<DynamicNodeInvocation(DynamicNodeInvocation const &)> const &) {
  NOT_IMPLEMENTED();
}

DynamicOpenDataflowGraph
  flatmap_dynamic_invocation_set(
    DynamicOpenDataflowGraph const &,
    std::function<std::unordered_set<DynamicNodeInvocation>(DynamicNodeInvocation const &)> const &) {
  NOT_IMPLEMENTED();
}

DynamicOpenDataflowGraph 
  dynamic_open_dataflow_graph_from_invocation_set(std::unordered_set<DynamicNodeInvocation> const &invocation_set) {

  std::unordered_set<DynamicValueAttrs> all_values = 
    flatmap(invocation_set,
            [](DynamicNodeInvocation const &invocation)
              -> std::unordered_set<DynamicValueAttrs>
            {
              return set_union(
                unordered_set_of(invocation.inputs),
                unordered_set_of(invocation.outputs));
            });

  ManyToOne<DynamicValueAttrs, DynamicNodeInvocation> value_to_producer;
  for (DynamicNodeInvocation const &invocation : invocation_set) {
    for (DynamicValueAttrs const &output : invocation.outputs) {
      value_to_producer.insert({output, invocation});
    }
  }
    
  std::unordered_set<DynamicValueAttrs> graph_inputs = 
    filter(all_values,
           [&](DynamicValueAttrs const &v) -> bool {
             return !value_to_producer.contains_l(v);
           });


  DynamicOpenDataflowGraph result = make_empty_dynamic_open_dataflow_graph();
  bidict<OpenDataflowValue, DynamicValueAttrs> value_map;

  for (DynamicValueAttrs const &graph_input : graph_inputs) {
    DataflowGraphInput added = result.raw.add_input(graph_input);
    value_map.equate(OpenDataflowValue{added}, graph_input);
  }

  auto inputs_have_been_added = [&](DynamicNodeInvocation const &invocation) -> bool {
    return all_of(
      invocation.inputs, 
      [&](DynamicValueAttrs const &input) -> bool {
        return value_map.contains_r(input);
      });
  };

  std::unordered_set<DynamicNodeInvocation> to_add = invocation_set;

  auto add_invocation_to_graph = [&](DynamicNodeInvocation const &invocation) -> void {
    NodeAddedResult added = result.raw.add_node(
          invocation.node_attrs,
          transform(invocation.inputs,
                    [&](DynamicValueAttrs const &input) -> OpenDataflowValue {
                      return value_map.at_r(input);
                    }),
          invocation.outputs);

    for (auto const &[invocation_output, graph_output] : zip_strict(invocation.outputs, added.outputs)) {
      value_map.equate(OpenDataflowValue{graph_output}, invocation_output);
    }

    to_add.erase(invocation);
  };

  auto add_next_invocation_to_graph = [&]() -> bool {
    for (DynamicNodeInvocation const &invocation : to_add) {
      if (inputs_have_been_added(invocation)) {
        add_invocation_to_graph(invocation);
        return true;
      }
    }

    return false;
  };

  do {
    ASSERT(add_next_invocation_to_graph());
  } while (to_add.size() > 0);

  return result;
}


} // namespace FlexFlow
