#include "task-spec/dynamic_graph/dynamic_open_dataflow_graph.h"
#include "utils/containers/flatmap.h"
#include "utils/containers/zip_strict.h"
#include "utils/graph/dataflow_graph/algorithms.h"
#include "utils/graph/instances/unordered_set_labelled_open_dataflow_graph.h"
#include "utils/graph/labelled_open_dataflow_graph/algorithms/find_isomorphism.h"
#include "utils/graph/node/algorithms.h"
#include "utils/graph/open_dataflow_graph/algorithms/get_inputs.h"
#include "utils/many_to_one/many_to_one.h"
#include "utils/containers/all_of.h"
#include "utils/containers/contains_duplicates.h"

namespace FlexFlow {

DynamicOpenDataflowGraph make_empty_dynamic_open_dataflow_graph() {
  return DynamicOpenDataflowGraph{
    std::unordered_set<DynamicNodeInvocation>{},
  };
}

bool full_dynamic_graph_satisfies(
  DynamicOpenDataflowGraph const &g,
  std::function<bool(DynamicNodeAttrs const &)> const &node_condition,
  std::function<bool(DynamicValueAttrs const &)> const &value_condition) {

  NOT_IMPLEMENTED();
  // return all_of(get_dynamic_nodes(g), node_condition) 
  //   && all_of(get_dynamic_values(g), value_condition);
}

bool no_part_of_dynamic_graph_satisfies(
  DynamicOpenDataflowGraph const &g,
  std::function<bool(DynamicNodeAttrs const &)> const &node_condition,
  std::function<bool(DynamicValueAttrs const &)> const &value_condition) {

  NOT_IMPLEMENTED();
  // return full_dynamic_graph_satisfies(
  //   g,
  //   [&](DynamicNodeAttrs const &n) -> bool {
  //     return !node_condition(n); 
  //   },
  //   [&](DynamicValueAttrs const &v) -> bool {
  //     return !value_condition(v); 
  //   });
}

std::unordered_multiset<DynamicNodeAttrs> get_dynamic_nodes(DynamicOpenDataflowGraph const &g) {
  NOT_IMPLEMENTED();
  // return transform(unordered_multiset_of(get_nodes(g.raw)), 
  //                  [&](Node const &n) -> DynamicNodeAttrs {
  //                    return g.raw.at(n);
  //                  });
}

std::unordered_multiset<DynamicValueAttrs> get_dynamic_values(DynamicOpenDataflowGraph const &g) {
  NOT_IMPLEMENTED();
  // return transform(unordered_multiset_of(get_open_dataflow_values(g.raw)), 
  //                  [&](OpenDataflowValue const &v) -> DynamicValueAttrs {
  //                    return g.raw.at(v);
  //                  });
}

std::unordered_set<DynamicNodeInvocation> get_dynamic_invocation_set(DynamicOpenDataflowGraph const &g) {
  // TODO(@lockshaw)(#pr): Not possible with named arguments
  NOT_IMPLEMENTED();
  // return transform(
  //   get_nodes(g.raw),
  //   [&](Node const &n) -> DynamicNodeInvocation {
  //     std::vector<OpenDataflowValue> n_inputs = get_inputs(g.raw, n);
  //     std::vector<DataflowOutput> n_outputs = get_outputs(g.raw, n);
  //
  //     std::vector<DynamicValueAttrs> inputs = 
  //       transform(n_inputs, 
  //                 [&](OpenDataflowValue const &v) -> DynamicValueAttrs {
  //                   return g.raw.at(v);
  //                 });
  //
  //     std::vector<DynamicValueAttrs> outputs = 
  //       transform(n_outputs,
  //                 [&](DataflowOutput const &v) -> DynamicValueAttrs {
  //                   return g.raw.at(OpenDataflowValue{v});
  //                 });
  //
  //     return DynamicNodeInvocation{
  //       /*inputs=*/inputs,
  //       /*node_attrs=*/g.raw.at(n),
  //       /*outptuts=*/outputs,
  //     };
  //   });
}

DynamicOpenDataflowGraph
  transform_dynamic_invocation_set(
    DynamicOpenDataflowGraph const &g,
    std::function<DynamicNodeInvocation(DynamicNodeInvocation const &)> const &f) {
  std::unordered_set<DynamicNodeInvocation> current_invocation_set = get_dynamic_invocation_set(g);
  std::unordered_set<DynamicNodeInvocation> new_invocation_set = 
    transform(current_invocation_set, f);

  return dynamic_open_dataflow_graph_from_invocation_set(new_invocation_set);
}

DynamicOpenDataflowGraph
  flatmap_dynamic_invocation_set(
    DynamicOpenDataflowGraph const &g,
    std::function<std::unordered_set<DynamicNodeInvocation>(DynamicNodeInvocation const &)> const &f) {

  std::unordered_set<DynamicNodeInvocation> current_invocation_set = get_dynamic_invocation_set(g);
  std::vector<DynamicNodeInvocation> new_invocation_set = 
    flatmap(vector_of(current_invocation_set), f);

  ASSERT(!contains_duplicates(new_invocation_set));

  return dynamic_open_dataflow_graph_from_invocation_set(unordered_set_of(new_invocation_set));
}

DynamicOpenDataflowGraph 
  dynamic_open_dataflow_graph_from_invocation_set(std::unordered_set<DynamicNodeInvocation> const &invocation_set) {

  // TODO(@lockshaw)(#pr): Not possible with named arguments
  NOT_IMPLEMENTED();
  // std::unordered_set<DynamicValueAttrs> all_values = 
  //   flatmap(invocation_set,
  //           [](DynamicNodeInvocation const &invocation)
  //             -> std::unordered_set<DynamicValueAttrs>
  //           {
  //             return set_union(
  //               unordered_set_of(invocation.inputs),
  //               unordered_set_of(invocation.outputs));
  //           });
  //
  // ManyToOne<DynamicValueAttrs, DynamicNodeInvocation> value_to_producer;
  // for (DynamicNodeInvocation const &invocation : invocation_set) {
  //   for (DynamicValueAttrs const &output : invocation.outputs) {
  //     value_to_producer.insert({output, invocation});
  //   }
  // }
  //
  // std::unordered_set<DynamicValueAttrs> graph_inputs = 
  //   filter(all_values,
  //          [&](DynamicValueAttrs const &v) -> bool {
  //            return !value_to_producer.contains_l(v);
  //          });
  //
  //
  // DynamicOpenDataflowGraph result = make_empty_dynamic_open_dataflow_graph();
  // bidict<OpenDataflowValue, DynamicValueAttrs> value_map;
  //
  // for (DynamicValueAttrs const &graph_input : graph_inputs) {
  //   DataflowGraphInput added = result.raw.add_input(graph_input);
  //   value_map.equate(OpenDataflowValue{added}, graph_input);
  // }
  //
  // auto inputs_have_been_added = [&](DynamicNodeInvocation const &invocation) -> bool {
  //   return all_of(
  //     invocation.inputs, 
  //     [&](DynamicValueAttrs const &input) -> bool {
  //       return value_map.contains_r(input);
  //     });
  // };
  //
  // std::unordered_set<DynamicNodeInvocation> to_add = invocation_set;
  //
  // auto add_invocation_to_graph = [&](DynamicNodeInvocation const &invocation) -> void {
  //   NodeAddedResult added = result.raw.add_node(
  //         invocation.node_attrs,
  //         transform(invocation.inputs,
  //                   [&](DynamicValueAttrs const &input) -> OpenDataflowValue {
  //                     return value_map.at_r(input);
  //                   }),
  //         invocation.outputs);
  //
  //   for (auto const &[invocation_output, graph_output] : zip_strict(invocation.outputs, added.outputs)) {
  //     value_map.equate(OpenDataflowValue{graph_output}, invocation_output);
  //   }
  //
  //   to_add.erase(invocation);
  // };
  //
  // auto add_next_invocation_to_graph = [&]() {
  //   for (DynamicNodeInvocation const &invocation : to_add) {
  //     if (inputs_have_been_added(invocation)) {
  //       add_invocation_to_graph(invocation);
  //       return;
  //     }
  //   }
  //
  //   PANIC("Failed to add any invocations in to_add", to_add);
  // };
  //
  // do {
  //   add_next_invocation_to_graph();
  // } while (to_add.size() > 0);
  //
  // ASSERT(get_dynamic_invocation_set(result) == invocation_set);
  //
  // return result;
}

bool dynamic_open_dataflow_graphs_are_isomorphic(DynamicOpenDataflowGraph const &lhs,
                                                 DynamicOpenDataflowGraph const &rhs) {
  NOT_IMPLEMENTED();
}

} // namespace FlexFlow
