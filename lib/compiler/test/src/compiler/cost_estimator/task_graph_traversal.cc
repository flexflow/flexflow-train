#include "compiler/cost_estimator/task_graph_traversal.h"
#include "compiler/cost_estimator/task_graph.dtg.h"
#include "compiler/cost_estimator/tasks_state_tracker.dtg.h"
#include "utils/graph/algorithms.h"
#include "utils/graph/digraph/directed_edge.dtg.h"
#include "utils/graph/instances/adjacency_digraph.h"
#include <doctest/doctest.h>
#include <optional>

namespace FlexFlow {

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("task_graph_traversal") {

    SUBCASE("linear graph") {
      DiGraph g = DiGraph::create<AdjacencyDiGraph>();
      std::vector<Node> n = add_nodes(g, 4);
      add_edges(g,
                {
                    DirectedEdge{n.at(0), n.at(1)},
                    DirectedEdge{n.at(1), n.at(2)},
                    DirectedEdge{n.at(2), n.at(3)},
                });

      std::unordered_map<Node, float> cost_map = {
          {n.at(0), 1}, {n.at(1), 10}, {n.at(2), 100}, {n.at(3), 1000}};
      auto is_allowed_to_run =
          [&](Node const &n, TasksStateTracker const &tracker) { return true; };

      TaskGraph task_graph = TaskGraph{g, cost_map, is_allowed_to_run};

      float result = simulate_forward_pass(task_graph);
      float correct = 1 + 10 + 100 + 1000;
      CHECK(correct == result);
    }

    SUBCASE("rhomboidal graph") {
      DiGraph g = DiGraph::create<AdjacencyDiGraph>();
      std::vector<Node> n = add_nodes(g, 4);

      add_edges(g,
                {DirectedEdge{n.at(0), n.at(1)},
                 DirectedEdge{n.at(0), n.at(2)},
                 DirectedEdge{n.at(1), n.at(3)},
                 DirectedEdge{n.at(2), n.at(3)}});

      std::unordered_map<Node, float> cost_map = {
          {n.at(0), 10}, {n.at(1), 15}, {n.at(2), 20}, {n.at(3), 25}};

      SUBCASE("no processing constraints") {
        auto is_allowed_to_run = [&](Node const &n,
                                     TasksStateTracker const &tracker) {
          return true;
        };

        TaskGraph task_graph = TaskGraph{g, cost_map, is_allowed_to_run};
        float result = simulate_forward_pass(task_graph);
        float correct = 10 + 20 + 25;
        CHECK(correct == result);
      }

      SUBCASE("processing constraint") {
        auto is_allowed_to_run = [&](Node const &n,
                                     TasksStateTracker const &tracker) {
          return tracker.tasks_processing.contents().size() == 0;
        };

        TaskGraph task_graph = TaskGraph{g, cost_map, is_allowed_to_run};
        float result = simulate_forward_pass(task_graph);
        float correct = 10 + 20 + 15 + 25;
        CHECK(correct == result);
      }
    }

    SUBCASE("diamond graph with crossing") {
      DiGraph g = DiGraph::create<AdjacencyDiGraph>();
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

      std::unordered_map<Node, float> cost_map = {{n.at(0), 10},
                                                  {n.at(1), 15},
                                                  {n.at(2), 20},
                                                  {n.at(3), 25},
                                                  {n.at(4), 30},
                                                  {n.at(5), 35}};

      SUBCASE("no processing constraints") {
        auto is_allowed_to_run = [&](Node const &n,
                                     TasksStateTracker const &tracker) {
          return true;
        };

        TaskGraph task_graph = TaskGraph{g, cost_map, is_allowed_to_run};
        float result = simulate_forward_pass(task_graph);
        float correct = 10 + std::max({15 + 25, 20 + 25, 20 + 30}) + 35;
        CHECK(correct == result);
      }

      SUBCASE("one node at a time") {
        auto is_allowed_to_run = [&](Node const &n,
                                     TasksStateTracker const &tracker) {
          return tracker.tasks_processing.contents().size() == 0;
        };

        TaskGraph task_graph = TaskGraph{g, cost_map, is_allowed_to_run};
        float result = simulate_forward_pass(task_graph);
        float correct = 10 + 15 + 20 + 25 + 30 + 35;
        CHECK(correct == result);
      }
    }

    SUBCASE("all-to-all intermediate") {
      DiGraph g = DiGraph::create<AdjacencyDiGraph>();
      std::vector<Node> n = add_nodes(g, 5);

      add_edges(g,
                {DirectedEdge{n.at(0), n.at(1)},
                 DirectedEdge{n.at(0), n.at(2)},
                 DirectedEdge{n.at(0), n.at(3)},
                 DirectedEdge{n.at(1), n.at(4)},
                 DirectedEdge{n.at(2), n.at(4)},
                 DirectedEdge{n.at(3), n.at(4)}});

      std::unordered_map<Node, float> cost_map = {{n.at(0), 10},
                                                  {n.at(1), 100},
                                                  {n.at(2), 100},
                                                  {n.at(3), 100},
                                                  {n.at(4), 20}};

      SUBCASE("at most two nodes at a time") {
        auto is_allowed_to_run = [&](Node const &n,
                                     TasksStateTracker const &tracker) {
          return tracker.tasks_processing.contents().size() < 2;
        };

        TaskGraph task_graph = TaskGraph{g, cost_map, is_allowed_to_run};
        float result = simulate_forward_pass(task_graph);
        float correct = 10 + (100 + 100) + 20;
        CHECK(correct == result);
      }
    }
  }
}
} // namespace FlexFlow
