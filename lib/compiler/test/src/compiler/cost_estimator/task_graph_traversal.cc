#include "compiler/cost_estimator/task_graph_traversal.h"
#include "compiler/cost_estimator/task_graph.dtg.h"
#include "compiler/cost_estimator/task_graph_profile.dtg.h"
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

      TaskGraphProfile result = simulate_forward_pass(task_graph);
      TaskGraphProfile correct =
          TaskGraphProfile{{
                               TaskProfile{n.at(0), 0, 1},
                               TaskProfile{n.at(1), 1, 11},
                               TaskProfile{n.at(2), 11, 111},
                               TaskProfile{n.at(3), 111, 1111},
                           },
                           1111};
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
        TaskGraphProfile result = simulate_forward_pass(task_graph);
        TaskGraphProfile correct =
            TaskGraphProfile{{
                                 TaskProfile{n.at(0), 0, 10},
                                 TaskProfile{n.at(1), 10, 25},
                                 TaskProfile{n.at(2), 10, 30},
                                 TaskProfile{n.at(3), 30, 55},
                             },
                             55};
        CHECK(correct == result);
      }

      SUBCASE("processing constraint") {
        auto is_allowed_to_run = [&](Node const &n,
                                     TasksStateTracker const &tracker) {
          return tracker.tasks_processing.contents().size() == 0;
        };

        TaskGraph task_graph = TaskGraph{g, cost_map, is_allowed_to_run};
        TaskGraphProfile result = simulate_forward_pass(task_graph);
        TaskGraphProfile correct =
            TaskGraphProfile{{
                                 TaskProfile{n.at(0), 0, 10},
                                 TaskProfile{n.at(1), 10, 25},
                                 TaskProfile{n.at(2), 25, 45},
                                 TaskProfile{n.at(3), 45, 70},
                             },
                             70};
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
        TaskGraphProfile result = simulate_forward_pass(task_graph);
        TaskGraphProfile correct =
            TaskGraphProfile{{
                                 TaskProfile{n.at(0), 0, 10},
                                 TaskProfile{n.at(1), 10, 25},
                                 TaskProfile{n.at(2), 10, 30},
                                 TaskProfile{n.at(3), 30, 55},
                                 TaskProfile{n.at(4), 30, 60},
                                 TaskProfile{n.at(5), 60, 95},
                             },
                             95};
        CHECK(correct == result);
      }

      SUBCASE("one node at a time") {
        auto is_allowed_to_run = [&](Node const &n,
                                     TasksStateTracker const &tracker) {
          return tracker.tasks_processing.contents().size() == 0;
        };

        TaskGraph task_graph = TaskGraph{g, cost_map, is_allowed_to_run};
        TaskGraphProfile result = simulate_forward_pass(task_graph);
        TaskGraphProfile correct =
            TaskGraphProfile{{
                                 TaskProfile{n.at(0), 0, 10},
                                 TaskProfile{n.at(1), 10, 25},
                                 TaskProfile{n.at(2), 25, 45},
                                 TaskProfile{n.at(3), 45, 70},
                                 TaskProfile{n.at(4), 70, 100},
                                 TaskProfile{n.at(5), 100, 135},
                             },
                             135};
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
        TaskGraphProfile result = simulate_forward_pass(task_graph);
        TaskGraphProfile correct =
            TaskGraphProfile{{
                                 TaskProfile{n.at(0), 0, 10},
                                 TaskProfile{n.at(1), 10, 110},
                                 TaskProfile{n.at(2), 10, 110},
                                 TaskProfile{n.at(3), 110, 210},
                                 TaskProfile{n.at(4), 210, 230},
                             },
                             230};
        CHECK(correct == result);
      }
    }
  }
}
} // namespace FlexFlow
