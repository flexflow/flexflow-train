#include "substitutions/unlabelled/find_pattern_matches.h"
#include "substitutions/unlabelled/match_additional_criterion.h"
#include "substitutions/unlabelled/pattern_matching.h"
#include "test/utils/doctest/fmt/vector.h"
#include "utils/containers/get_only.h"
#include "utils/containers/make_counter_func.h"
#include "utils/containers/require_only_key.h"
#include "utils/graph/instances/unordered_set_open_kwarg_dataflow_graph.h"
#include "utils/graph/node/algorithms.h"
#include "utils/graph/open_kwarg_dataflow_graph/algorithms/get_all_open_kwarg_dataflow_values.h"
#include "utils/graph/open_kwarg_dataflow_graph/algorithms/get_incoming_open_kwarg_dataflow_edges_for_node.h"
#include "utils/graph/open_kwarg_dataflow_graph/algorithms/get_open_kwarg_dataflow_graph_subgraph.h"
#include "utils/graph/open_kwarg_dataflow_graph/algorithms/get_open_kwarg_dataflow_subgraph_inputs.h"
#include "utils/graph/open_kwarg_dataflow_graph/open_kwarg_dataflow_graph.h"
#include <doctest/doctest.h>

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("find_pattern_matches") {
    OpenKwargDataflowGraph<int, TensorSlotName> pattern_graph =
        OpenKwargDataflowGraph<int, TensorSlotName>::create<
            UnorderedSetOpenKwargDataflowGraph<int, TensorSlotName>>();

    KwargNodeAddedResult pattern_n0_added = pattern_graph.add_node(
        /*inputs=*/{},
        /*outputs=*/{
            TensorSlotName::OUTPUT,
        });
    Node pattern_n0 = pattern_n0_added.node;
    OpenKwargDataflowValue<int, TensorSlotName> pattern_v0 =
        OpenKwargDataflowValue<int, TensorSlotName>{
            require_only_key(pattern_n0_added.outputs, TensorSlotName::OUTPUT),
        };

    KwargNodeAddedResult pattern_n1_added = pattern_graph.add_node(
        /*inputs=*/
        {
            {
                TensorSlotName::INPUT,
                pattern_v0,
            },
        },
        /*outputs=*/{TensorSlotName::OUTPUT});
    Node pattern_n1 = pattern_n1_added.node;
    OpenKwargDataflowValue<int, TensorSlotName> pattern_v1 =
        OpenKwargDataflowValue<int, TensorSlotName>{
            require_only_key(pattern_n1_added.outputs, TensorSlotName::OUTPUT),
        };

    UnlabelledGraphPattern pattern = UnlabelledGraphPattern{pattern_graph};
    PatternNode p0 = PatternNode{pattern_n0};
    PatternNode p1 = PatternNode{pattern_n1};

    OpenKwargDataflowGraph<int, TensorSlotName> graph =
        OpenKwargDataflowGraph<int, TensorSlotName>::create<
            UnorderedSetOpenKwargDataflowGraph<int, TensorSlotName>>();

    KwargNodeAddedResult n0_added = graph.add_node(
        /*inputs=*/{},
        /*outputs=*/{
            TensorSlotName::OUTPUT,
        });
    Node n0 = n0_added.node;
    OpenKwargDataflowValue<int, TensorSlotName> v0 =
        OpenKwargDataflowValue<int, TensorSlotName>{
            require_only_key(n0_added.outputs, TensorSlotName::OUTPUT),
        };

    KwargNodeAddedResult n1_added = graph.add_node(
        /*inputs=*/
        {
            {
                TensorSlotName::INPUT,
                v0,
            },
        },
        /*outputs=*/{
            TensorSlotName::OUTPUT,
        });
    Node n1 = n1_added.node;
    OpenKwargDataflowValue<int, TensorSlotName> v1 =
        OpenKwargDataflowValue<int, TensorSlotName>{
            require_only_key(n1_added.outputs, TensorSlotName::OUTPUT),
        };

    KwargNodeAddedResult n2_added = graph.add_node(
        /*inputs=*/
        {
            {
                TensorSlotName::INPUT,
                v1,
            },
        },
        /*outputs=*/{
            TensorSlotName::OUTPUT,
        });
    Node n2 = n2_added.node;
    OpenKwargDataflowValue<int, TensorSlotName> v2 =
        OpenKwargDataflowValue<int, TensorSlotName>{
            require_only_key(n2_added.outputs, TensorSlotName::OUTPUT),
        };

    KwargNodeAddedResult n3_added = graph.add_node(
        /*inputs=*/
        {
            {
                TensorSlotName::INPUT,
                v2,
            },
        },
        /*outputs=*/{
            TensorSlotName::OUTPUT,
        });
    Node n3 = n3_added.node;
    OpenKwargDataflowValue<int, TensorSlotName> v3 =
        OpenKwargDataflowValue<int, TensorSlotName>{
            require_only_key(n3_added.outputs, TensorSlotName::OUTPUT),
        };

    UnlabelledKwargDataflowGraphPatternMatch match =
        UnlabelledKwargDataflowGraphPatternMatch{
            bidict<PatternNode, Node>{
                {p0, n0},
                {p1, n1},
            },
            bidict<PatternInput,
                   OpenKwargDataflowValue<int, TensorSlotName>>{}};

    UnlabelledKwargDataflowGraphPatternMatch invalid_match =
        UnlabelledKwargDataflowGraphPatternMatch{
            bidict<PatternNode, Node>{
                {p0, n1},
                {p1, n2},
            },
            bidict<PatternInput,
                   OpenKwargDataflowValue<int, TensorSlotName>>{}};

    std::unordered_map<TensorSlotName,
                       OpenKwargDataflowEdge<int, TensorSlotName>>
        n1_incoming = {
            {
                TensorSlotName::INPUT,
                OpenKwargDataflowEdge<int, TensorSlotName>{
                    KwargDataflowEdge<TensorSlotName>{
                        KwargDataflowOutput{n0, TensorSlotName::OUTPUT},
                        KwargDataflowInput{n1, TensorSlotName::INPUT},
                    },
                },
            },
        };

    SUBCASE("get_incoming_edges") {
      SUBCASE("n0") {
        std::unordered_map<TensorSlotName,
                           OpenKwargDataflowEdge<int, TensorSlotName>>
            result = get_incoming_open_kwarg_dataflow_edges_for_node(graph, n0);
        std::unordered_map<TensorSlotName,
                           OpenKwargDataflowEdge<int, TensorSlotName>>
            correct = {};
        CHECK(result == correct);
      }

      SUBCASE("n1") {
        std::unordered_map<TensorSlotName,
                           OpenKwargDataflowEdge<int, TensorSlotName>>
            result = get_incoming_open_kwarg_dataflow_edges_for_node(graph, n1);
        std::unordered_map<TensorSlotName,
                           OpenKwargDataflowEdge<int, TensorSlotName>>
            correct = n1_incoming;
        CHECK(result == correct);
      }
    }

    SUBCASE("get_open_kwarg_dataflow_subgraph_inputs") {
      std::unordered_set<OpenKwargDataflowValue<int, TensorSlotName>> result =
          get_open_kwarg_dataflow_subgraph_inputs(graph, {n0, n1});
      std::unordered_set<OpenKwargDataflowValue<int, TensorSlotName>> correct =
          {};
      CHECK(result == correct);
    }

    SUBCASE("get_open_kwarg_dataflow_graph_subgraph") {
      int graph_input_ctr = 0;
      OpenKwargDataflowGraphView<int, TensorSlotName> g =
          get_open_kwarg_dataflow_graph_subgraph(
              graph, {n0, n1}, make_counter_func())
              .graph;

      SUBCASE("nodes") {
        std::unordered_set<Node> result = get_nodes(g);
        std::unordered_set<Node> correct = {n0, n1};
        CHECK(result == correct);
      }

      SUBCASE("inputs") {
        std::unordered_set<KwargDataflowGraphInput<int>> result =
            g.get_inputs();
        std::unordered_set<KwargDataflowGraphInput<int>> correct = {};
        CHECK(result == correct);
      }

      SUBCASE("get_all_open_kwarg_dataflow_values") {
        std::unordered_set<OpenKwargDataflowValue<int, TensorSlotName>> values =
            get_all_open_kwarg_dataflow_values(g);
        CHECK(values.size() == 2);
      }
    }

    SUBCASE("subgraph_matched") {
      OpenKwargDataflowGraphView<int, TensorSlotName> result =
          subgraph_matched(graph, match).graph;
      std::unordered_set<Node> result_nodes = get_nodes(result);
      std::unordered_set<Node> correct_nodes = {n0, n1};
      CHECK(result_nodes == correct_nodes);
    }

    SUBCASE("unlabelled_pattern_does_match") {
      CHECK(unlabelled_pattern_does_match(
          pattern, graph, match, match_additional_crition_always_true()));
      CHECK_FALSE(unlabelled_pattern_does_match(
          pattern,
          graph,
          invalid_match,
          match_additional_crition_always_true()));
    }

    SUBCASE("unlabelled_pattern_does_match (open)") {
      OpenKwargDataflowGraph<int, TensorSlotName> g =
          OpenKwargDataflowGraph<int, TensorSlotName>::create<
              UnorderedSetOpenKwargDataflowGraph<int, TensorSlotName>>();
      KwargDataflowGraphInput<int> i0 = g.add_input(0);

      KwargNodeAddedResult g_n0_added = g.add_node(
          /*inputs=*/
          {
              {
                  TensorSlotName::INPUT,
                  OpenKwargDataflowValue<int, TensorSlotName>{i0},
              },
          },
          /*outputs=*/{
              TensorSlotName::OUTPUT,
          });
      Node g_n0 = g_n0_added.node;
      OpenKwargDataflowValue<int, TensorSlotName> g_v0 =
          OpenKwargDataflowValue<int, TensorSlotName>{
              require_only_key(g_n0_added.outputs, TensorSlotName::OUTPUT),
          };
      PatternNode g_p0 = PatternNode{g_n0};
      PatternInput g_pi0 = PatternInput{i0};

      UnlabelledGraphPattern open_pattern = UnlabelledGraphPattern{g};

      UnlabelledKwargDataflowGraphPatternMatch open_match =
          UnlabelledKwargDataflowGraphPatternMatch{
              bidict<PatternNode, Node>{
                  {g_p0, n1},
              },
              bidict<PatternInput, OpenKwargDataflowValue<int, TensorSlotName>>{
                  {g_pi0, v0},
              }};
      CHECK(unlabelled_pattern_does_match(
          open_pattern,
          graph,
          open_match,
          match_additional_crition_always_true()));
    }

    SUBCASE("find_pattern_matches") {
      std::vector<UnlabelledKwargDataflowGraphPatternMatch> matches =
          find_unlabelled_pattern_matches(
              pattern, graph, match_additional_crition_always_true());
      std::vector<UnlabelledKwargDataflowGraphPatternMatch> correct = {match};

      CHECK(matches == correct);
    }
  }
}
