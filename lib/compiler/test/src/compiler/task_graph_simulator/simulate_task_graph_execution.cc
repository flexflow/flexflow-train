#include "compiler/task_graph_simulator/simulate_task_graph_execution.h"
#include "compiler/task_graph_simulator/task_graph_execution_state.dtg.h"
#include "compiler/task_graph_simulator/task_graph_execution_trace.dtg.h"
#include "utils/containers/lookup_in_map.h"
#include "utils/graph/algorithms.h"
#include "utils/graph/digraph/directed_edge.dtg.h"
#include "utils/graph/instances/adjacency_digraph.h"
#include <doctest/doctest.h>
#include <optional>

namespace FlexFlow {

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("simulate_task_graph_execution") {
    DiGraph g = DiGraph::create<AdjacencyDiGraph>();
    SUBCASE("linear graph") {
      std::vector<Node> n = add_nodes(g, 4);
      add_edges(g,
                {
                    DirectedEdge{n.at(0), n.at(1)},
                    DirectedEdge{n.at(1), n.at(2)},
                    DirectedEdge{n.at(2), n.at(3)},
                });

      auto cost_function = lookup_in_map<Node, float>(
          {{n.at(0), 1}, {n.at(1), 10}, {n.at(2), 100}, {n.at(3), 1000}});

      auto is_allowed_to_run =
          [&](Node const &n,
              std::unordered_set<Node> const &in_progress_tasks,
              std::unordered_set<Node> const &finished_tasks) { return true; };

      TaskExecutionConstraint constraint =
          TaskExecutionConstraint{is_allowed_to_run};

      TaskGraphExecutionTrace result =
          simulate_task_graph_execution(g, cost_function, constraint);
      TaskGraphExecutionTrace correct = TaskGraphExecutionTrace{{
          TaskProfile{n.at(0), 0, 1},
          TaskProfile{n.at(1), 1, 11},
          TaskProfile{n.at(2), 11, 111},
          TaskProfile{n.at(3), 111, 1111},
      }};
      CHECK(correct == result);
    }

    SUBCASE("rhomboidal graph") {
      std::vector<Node> n = add_nodes(g, 4);

      add_edges(g,
                {DirectedEdge{n.at(0), n.at(1)},
                 DirectedEdge{n.at(0), n.at(2)},
                 DirectedEdge{n.at(1), n.at(3)},
                 DirectedEdge{n.at(2), n.at(3)}});

      auto cost_function = lookup_in_map<Node, float>(
          {{n.at(0), 10}, {n.at(1), 15}, {n.at(2), 20}, {n.at(3), 25}});

      SUBCASE("no processing constraints") {
        auto is_allowed_to_run =
            [&](Node const &n,
                std::unordered_set<Node> const &in_progress_tasks,
                std::unordered_set<Node> const &finished_tasks) {
              return true;
            };

        TaskExecutionConstraint constraint =
            TaskExecutionConstraint{is_allowed_to_run};
        TaskGraphExecutionTrace result =
            simulate_task_graph_execution(g, cost_function, constraint);
        TaskGraphExecutionTrace correct = TaskGraphExecutionTrace{{
            TaskProfile{n.at(0), 0, 10},
            TaskProfile{n.at(1), 10, 25},
            TaskProfile{n.at(2), 10, 30},
            TaskProfile{n.at(3), 30, 55},
        }};
        CHECK(correct == result);
      }

      SUBCASE("one node at a time") {
        auto is_allowed_to_run =
            [&](Node const &n,
                std::unordered_set<Node> const &in_progress_tasks,
                std::unordered_set<Node> const &finished_tasks) {
              return in_progress_tasks.size() == 0;
            };

        TaskExecutionConstraint constraint =
            TaskExecutionConstraint{is_allowed_to_run};
        TaskGraphExecutionTrace result =
            simulate_task_graph_execution(g, cost_function, constraint);
        TaskGraphExecutionTrace correct = TaskGraphExecutionTrace{{
            TaskProfile{n.at(0), 0, 10},
            TaskProfile{n.at(1), 10, 25},
            TaskProfile{n.at(2), 25, 45},
            TaskProfile{n.at(3), 45, 70},
        }};
        CHECK(correct == result);
      }
    }

    SUBCASE("diamond graph with crossing") {
      std::vector<Node> n = add_nodes(g, 6);

      add_edges(g,
                {
                    DirectedEdge{n.at(0), n.at(1)},
                    DirectedEdge{n.at(0), n.at(2)},
                    DirectedEdge{n.at(1), n.at(3)},
                    DirectedEdge{n.at(2), n.at(3)},
                    DirectedEdge{n.at(2), n.at(4)},
                    DirectedEdge{n.at(3), n.at(5)},
                    DirectedEdge{n.at(4), n.at(5)},
                });

      auto cost_function = lookup_in_map<Node, float>({{n.at(0), 10},
                                                       {n.at(1), 15},
                                                       {n.at(2), 20},
                                                       {n.at(3), 25},
                                                       {n.at(4), 30},
                                                       {n.at(5), 35}});

      SUBCASE("no processing constraints") {
        auto is_allowed_to_run =
            [&](Node const &n,
                std::unordered_set<Node> const &in_progress_tasks,
                std::unordered_set<Node> const &finished_tasks) {
              return true;
            };

        TaskExecutionConstraint constraint =
            TaskExecutionConstraint{is_allowed_to_run};
        TaskGraphExecutionTrace result =
            simulate_task_graph_execution(g, cost_function, constraint);
        TaskGraphExecutionTrace correct = TaskGraphExecutionTrace{{
            TaskProfile{n.at(0), 0, 10},
            TaskProfile{n.at(1), 10, 25},
            TaskProfile{n.at(2), 10, 30},
            TaskProfile{n.at(3), 30, 55},
            TaskProfile{n.at(4), 30, 60},
            TaskProfile{n.at(5), 60, 95},
        }};
        CHECK(correct == result);
      }

      SUBCASE("one node at a time") {
        auto is_allowed_to_run =
            [&](Node const &n,
                std::unordered_set<Node> const &in_progress_tasks,
                std::unordered_set<Node> const &finished_tasks) {
              return in_progress_tasks.size() == 0;
            };

        TaskExecutionConstraint constraint =
            TaskExecutionConstraint{is_allowed_to_run};
        TaskGraphExecutionTrace result =
            simulate_task_graph_execution(g, cost_function, constraint);
        TaskGraphExecutionTrace correct = TaskGraphExecutionTrace{{
            TaskProfile{n.at(0), 0, 10},
            TaskProfile{n.at(1), 10, 25},
            TaskProfile{n.at(2), 25, 45},
            TaskProfile{n.at(3), 45, 70},
            TaskProfile{n.at(4), 70, 100},
            TaskProfile{n.at(5), 100, 135},
        }};
        CHECK(correct == result);
      }
    }

    SUBCASE("all-to-all intermediate") {
      std::vector<Node> n = add_nodes(g, 5);

      add_edges(g,
                {DirectedEdge{n.at(0), n.at(1)},
                 DirectedEdge{n.at(0), n.at(2)},
                 DirectedEdge{n.at(0), n.at(3)},
                 DirectedEdge{n.at(1), n.at(4)},
                 DirectedEdge{n.at(2), n.at(4)},
                 DirectedEdge{n.at(3), n.at(4)}});

      auto cost_function = lookup_in_map<Node, float>({{n.at(0), 10},
                                                       {n.at(1), 100},
                                                       {n.at(2), 100},
                                                       {n.at(3), 100},
                                                       {n.at(4), 20}});

      SUBCASE("at most two nodes at a time") {
        auto is_allowed_to_run =
            [&](Node const &n,
                std::unordered_set<Node> const &in_progress_tasks,
                std::unordered_set<Node> const &finished_tasks) {
              return in_progress_tasks.size() < 2;
            };

        TaskExecutionConstraint constraint =
            TaskExecutionConstraint{is_allowed_to_run};
        TaskGraphExecutionTrace result =
            simulate_task_graph_execution(g, cost_function, constraint);
        TaskGraphExecutionTrace correct = TaskGraphExecutionTrace{{
            TaskProfile{n.at(0), 0, 10},
            TaskProfile{n.at(1), 10, 110},
            TaskProfile{n.at(2), 10, 110},
            TaskProfile{n.at(3), 110, 210},
            TaskProfile{n.at(4), 210, 230},
        }};
        CHECK(correct == result);
      }
    }
  }
}
} // namespace FlexFlow
