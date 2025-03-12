#include "compiler/mcmc/mcmc_algorithm.h"
#include "compiler/machine_mapping/allowed_machine_views.h"
#include "compiler/mcmc/machine_mapping_mutation_set.h"
#include "compiler/mcmc/mcmc_graph_optimize_state.h"
#include "compiler/task_graph_simulator/task_simulator.h"
#include "pcg/operator_task_space.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph_edge.h"
#include "pcg/parallel_computation_graph/parallel_tensor_guid_t.h"
#include "substitutions/apply_substitution/apply_substitution.h"
#include "substitutions/apply_substitution/evaluate_substitution_output.h"
#include "substitutions/apply_substitution/output_expr_to_result_sub_pcg_mapping.h"
#include "substitutions/open_parallel_tensor_guid_t.h"
#include "substitutions/pcg_pattern.h"
#include "substitutions/pcg_pattern_match.h"
#include "substitutions/sub_parallel_computation_graph.h"
#include "substitutions/sub_parallel_computation_graph_data.dtg.h"
#include "substitutions/sub_parallel_computation_graph_edge.h"
#include "substitutions/substitution.h"
#include "substitutions/unity_substitution_set.h"
#include "utils/containers/keys.h"
#include "utils/containers/merge_maps.h"
#include "utils/containers/restrict_keys.h"
#include "utils/containers/set_minus.h"
#include "utils/containers/transform.h"
#include "utils/containers/values.h"
#include "utils/deduplicated_priority_queue.h"
#include "utils/full_binary_tree/binary_tree_path.h"
#include "utils/graph/node/algorithms.h"
#include "utils/optional.h"

namespace FlexFlow {

std::optional<MachineMapping>
    get_naive_mapping(ParallelComputationGraph &pcg,
                      MachineSpecification const &resources) {
  std::vector<parallel_layer_guid_t> layers = topological_ordering(pcg);
  std::unordered_map<parallel_layer_guid_t, MachineView> machine_views;
  for (parallel_layer_guid_t layer : layers) {
    OperatorTaskSpace task = get_operator_task_space(pcg, layer);
    std::unordered_set<MachineView> allowed_machine_views =
        get_allowed_machine_views(resources, task, DeviceType::GPU);
    if (allowed_machine_views.empty()) {
      return std::nullopt;
    }
    machine_views.insert({layer, *(allowed_machine_views.begin())});
  }
  return MachineMapping{machine_views};
}

SearchResult apply_substitution_and_update_machine_mapping(
    SearchResult const &mapped_pcg,
    Substitution const &sub,
    PCGPatternMatch const &match) {
  // std::cout << "applying substitution" << std::endl;
  SubParallelComputationGraph spcg = sub_pcg_from_full_pcg(mapped_pcg.pcg);

  auto substitution_output_result =
      evaluate_substitution_output(spcg, sub, match);
  SubParallelComputationGraph substitution_output_graph =
      substitution_output_result.first;
  OutputExprToResultSubPCGMapping output_expr_to_result_sub_pcg_mapping =
      substitution_output_result.second;

  SubParallelComputationGraphData output_graph_data =
      get_sub_pcg_data(substitution_output_graph);
  SubParallelComputationGraphData pre_data = get_sub_pcg_data(spcg);

  std::unordered_set<parallel_layer_guid_t> pre_nodes =
      keys(pre_data.node_data);
  std::unordered_set<parallel_layer_guid_t> matched_nodes =
      unordered_set_of(values(match.node_assignment));
  std::unordered_set<parallel_layer_guid_t> post_nodes_from_original_graph =
      set_minus(pre_nodes, matched_nodes);

  std::unordered_map<parallel_layer_guid_t, MachineView> machine_views =
      mapped_pcg.machine_mapping.machine_views;

  std::unordered_set<MachineView> substituted_machine_views =
      transform(matched_nodes, [&](parallel_layer_guid_t const &node) {
        return machine_views.at(node);
      });

  std::unordered_map<parallel_layer_guid_t, ParallelLayerAttrs> post_node_data =
      [&] {
        std::unordered_map<parallel_layer_guid_t, ParallelLayerAttrs>
            post_node_data_from_orig = restrict_keys(
                pre_data.node_data, post_nodes_from_original_graph);
        std::unordered_map<parallel_layer_guid_t, ParallelLayerAttrs>
            post_node_data_from_sub = output_graph_data.node_data;

        // just taking the first substituted machine view, not sure if this
        // is fine
        for (auto [layer, attrs] : post_node_data_from_sub) {
          machine_views.try_emplace(layer, *substituted_machine_views.begin());
        }

        return merge_disjoint_maps(post_node_data_from_orig,
                                   post_node_data_from_sub);
      }();

  std::unordered_set<SubParallelComputationGraphEdge> post_edges = [&] {
    std::unordered_set<SubParallelComputationGraphEdge> post_edges_from_orig =
        filter(pre_data.edges, [&](SubParallelComputationGraphEdge const &e) {
          if (e.raw_edge.has<DataflowInputEdge>()) {
            return true;
          } else {
            DataflowEdge dfe = e.raw_edge.get<DataflowEdge>();
            parallel_layer_guid_t src = parallel_layer_guid_t{dfe.src.node};
            parallel_layer_guid_t dst = parallel_layer_guid_t{dfe.dst.node};
            return !(contains(matched_nodes, src) ||
                     contains(matched_nodes, dst));
          }
        });

    std::unordered_set<SubParallelComputationGraphEdge> post_edges_from_sub =
        filter(output_graph_data.edges,
               [&](SubParallelComputationGraphEdge const &e) {
                 return !e.raw_edge.has<DataflowInputEdge>();
               });

    bidict<PatternNodeOutput, parallel_tensor_guid_t>
        output_orig_pattern_mapping = get_output_mapping_for_pcg_pattern_match(
            match, sub.pcg_pattern, spcg);
    bidict<parallel_tensor_guid_t, OutputGraphExprNodeOutput>
        output_post_outexpr_mapping = get_output_graph_expr_output_mapping(
            output_expr_to_result_sub_pcg_mapping,
            sub.output_graph_expr,
            substitution_output_graph);

    std::unordered_set<SubParallelComputationGraphEdge> incoming_to_sub_edges;
    for (auto const &[pattern_input, base_graph_tensor] :
         match.input_assignment) {
      OutputGraphExprInput output_expr_input =
          sub.inputs_mapping.at_l(pattern_input);
      input_parallel_tensor_guid_t output_graph_input =
          output_expr_to_result_sub_pcg_mapping.input_mapping.at_r(
              output_expr_input);
      std::unordered_set<parallel_tensor_use_t> uses = get_parallel_tensor_uses(
          substitution_output_graph,
          open_parallel_tensor_guid_from_input(output_graph_input));
      for (parallel_tensor_use_t const &use : uses) {
        SubParallelComputationGraphEdge new_edge =
            subpcg_edge_from_tensor_and_use(base_graph_tensor, use);
        incoming_to_sub_edges.insert(new_edge);
      }
    }

    std::unordered_set<SubParallelComputationGraphEdge> outgoing_from_sub_edges;
    for (ParallelComputationGraphEdge const &outgoing_edge :
         get_subgraph_outgoing_edges(spcg, matched_nodes)) {
      parallel_tensor_guid_t original_tensor =
          get_parallel_tensor(outgoing_edge);
      PatternNodeOutput pattern_tensor =
          output_orig_pattern_mapping.at_r(original_tensor);
      OutputGraphExprNodeOutput output_graph_tensor =
          sub.outputs_mapping.at_l(pattern_tensor);
      parallel_tensor_guid_t new_tensor =
          output_post_outexpr_mapping.at_r(output_graph_tensor);

      SubParallelComputationGraphEdge new_edge =
          subpcg_edge_from_tensor_and_dst(
              new_tensor,
              get_dst_layer(outgoing_edge),
              get_dst_layer_input_idx(outgoing_edge));
      outgoing_from_sub_edges.insert(new_edge);
    }

    return set_union(std::vector{
        post_edges_from_orig,
        post_edges_from_sub,
        incoming_to_sub_edges,
        outgoing_from_sub_edges,
    });
  }();

  std::unordered_set<input_parallel_tensor_guid_t> post_inputs =
      pre_data.inputs;

  std::unordered_map<open_parallel_tensor_guid_t, ParallelTensorAttrs>
      post_value_data = [&] {
        std::unordered_map<open_parallel_tensor_guid_t, ParallelTensorAttrs>
            post_value_data_from_orig = filter_keys(
                pre_data.value_data, [&](open_parallel_tensor_guid_t const &t) {
                  return visit_open_parallel_tensor_guid(
                      t,
                      overload{
                          [&](parallel_tensor_guid_t const &t) {
                            return contains(post_nodes_from_original_graph,
                                            get_source_layer(t));
                          },
                          [](input_parallel_tensor_guid_t const &) {
                            return true;
                          },
                      });
                });

        std::unordered_map<open_parallel_tensor_guid_t, ParallelTensorAttrs>
            post_value_data_from_sub = output_graph_data.value_data;
        return merge_disjoint_maps(post_value_data_from_orig,
                                   post_value_data_from_sub);
      }();

  SubParallelComputationGraphData post_data = SubParallelComputationGraphData{
      post_node_data,
      post_edges,
      post_inputs,
      post_value_data,
  };

  return SearchResult{
      pcg_from_sub_pcg_by_dropping_inputs(sub_pcg_from_graph_data(post_data)),
      MachineMapping{machine_views}};
}

std::vector<SearchResult> all_pcgs_obtained_by_applying_a_substitution(
    SearchResult const &mapped_pcg,
    std::vector<Substitution> const &substitutions) {
  std::vector<SearchResult> results;
  SubParallelComputationGraph subpcg = sub_pcg_from_full_pcg(mapped_pcg.pcg);
  // std::cout << "len" << substitutions.size() << std::endl;
  for (Substitution const &substitution : substitutions) {
     std::cout << "in outer loop" << std::endl;
    for (PCGPatternMatch const &pattern_match :
         find_pattern_matches(substitution.pcg_pattern, subpcg)) {
       std::cout << "getting stuff" << std::endl;
      SearchResult mapped_pcg_from_substitution =
          apply_substitution_and_update_machine_mapping(
              mapped_pcg, substitution, pattern_match);
      results.push_back(mapped_pcg_from_substitution);
    }
  }
  return results;
}

SearchResult mcmc_graph_optimize(ParallelComputationGraph &pcg,
                                 CostEstimator const &cost_estimator,
                                 MachineSpecification const &resources,
                                 UnitySearchConfig const &search_config) {

  std::vector<Substitution> substitutions = get_substitution_set(resources);
  DeduplicatedPriorityQueue<MCMCOptimizeState> candidates;

  std::optional<MachineMapping> naive_mapping =
      get_naive_mapping(pcg, resources);
  if (naive_mapping == std::nullopt) {
    throw std::runtime_error("Failed to find any solutions");
  }

  // multiply runtime by -1 to make it minheap instead of maxheap
  MCMCOptimizeState best_state = MCMCOptimizeState{
      SearchResult{pcg, naive_mapping.value()},
      -1 * task_simulator_estimate_forward_pass_time(
               pcg, cost_estimator, naive_mapping.value(), resources)};

  candidates.push(best_state);

  for (int iteration = 0;
       !candidates.empty() && iteration < search_config.budget;
       ++iteration) {
    MCMCOptimizeState current_state = candidates.top();
    candidates.pop();

    SearchResult current_mapped_pcg = current_state.mapped_pcg;
    float current_estimate = current_state.runtime * -1;
    float best_estimate = best_state.runtime * -1;

    if (current_estimate < best_estimate) {
      best_state = current_state;
      std::cout << "new best state" << std::endl;
      std::cout << current_estimate << std::endl;
      std::cout << best_estimate << std::endl;
    } else if (current_estimate > best_estimate * search_config.alpha) {
      continue;
    } else {
      std::cout << current_estimate << best_estimate * search_config.alpha
                << std::endl;
    }
    // std::cout << "Hello" << std::endl;

    for (SearchResult const &new_mapped_pcg :
         all_pcgs_obtained_by_applying_a_substitution(current_mapped_pcg,
                                                      substitutions)) {
      float new_estimate = task_simulator_estimate_forward_pass_time(
          new_mapped_pcg.pcg,
          cost_estimator,
          new_mapped_pcg.machine_mapping,
          resources);

      std::cout << "new substitution" << std::endl;

      std::cout << "new estimate" << new_estimate << std::endl;
      if (new_estimate <= search_config.threshold &&
          get_nodes(new_mapped_pcg.pcg.raw_graph).size() <=
              search_config.max_num_ops) {
        candidates.push(MCMCOptimizeState{new_mapped_pcg, -1 * new_estimate});
      }
    }

    for (MachineMapping const &new_machine_mapping :
         get_possible_mutations(current_mapped_pcg, resources)) {
      float new_estimate =
          task_simulator_estimate_forward_pass_time(current_mapped_pcg.pcg,
                                                    cost_estimator,
                                                    new_machine_mapping,
                                                    resources);
      //std::cout << "new mapping" << std::endl;

      //std::cout << "new estimate" << new_estimate << std::endl;
      if (new_estimate <= search_config.threshold) {
        //std::cout << "pushed" << std::endl;
        candidates.push(
            MCMCOptimizeState{SearchResult{current_mapped_pcg.pcg, new_machine_mapping}, -1 * new_estimate});
      }
    }
  }
  return best_state.mapped_pcg;
}

} // namespace FlexFlow
