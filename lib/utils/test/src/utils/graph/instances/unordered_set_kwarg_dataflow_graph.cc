#include "utils/graph/instances/unordered_set_kwarg_dataflow_graph.h"
#include "utils/graph/kwarg_dataflow_graph/kwarg_dataflow_edge_query.h"
#include "utils/graph/kwarg_dataflow_graph/kwarg_dataflow_graph.h"
#include "utils/graph/kwarg_dataflow_graph/kwarg_dataflow_output_query.h"
#include "utils/graph/node/node_query.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("UnorderedSetKwargDataflowGraph") {
    KwargDataflowGraph<std::string> g = KwargDataflowGraph<std::string>::create<
        UnorderedSetKwargDataflowGraph<std::string>>();

    {
      std::unordered_set<Node> result = g.query_nodes(node_query_all());
      std::unordered_set<Node> correct = {};
      REQUIRE(result == correct);
    }

    {
      std::unordered_set<KwargDataflowEdge<std::string>> result =
          g.query_edges(kwarg_dataflow_edge_query_all<std::string>());
      std::unordered_set<KwargDataflowEdge<std::string>> correct = {};
      REQUIRE(result == correct);
    }

    {
      std::unordered_set<KwargDataflowOutput<std::string>> result =
          g.query_outputs(kwarg_dataflow_output_query_all<std::string>());
      std::unordered_set<KwargDataflowOutput<std::string>> correct = {};
      REQUIRE(result == correct);
    }

    KwargNodeAddedResult<std::string> added = g.add_node(
        /*inputs=*/{},
        /*output_slots=*/{
            "output_1",
            "output_2",
            "output_3",
        });

    KwargDataflowOutput<std::string> added_output_1 =
        added.outputs.at("output_1");

    KwargDataflowOutput<std::string> added_output_2 =
        added.outputs.at("output_2");

    KwargDataflowOutput<std::string> added_output_3 =
        added.outputs.at("output_3");

    {
      std::unordered_set<Node> result = g.query_nodes(node_query_all());
      std::unordered_set<Node> correct = {added.node};
      REQUIRE(result == correct);
    }

    {
      std::unordered_set<KwargDataflowEdge<std::string>> result =
          g.query_edges(kwarg_dataflow_edge_query_all<std::string>());
      std::unordered_set<KwargDataflowEdge<std::string>> correct = {};
      REQUIRE(result == correct);
    }

    {
      std::unordered_set<KwargDataflowOutput<std::string>> result =
          g.query_outputs(kwarg_dataflow_output_query_all<std::string>());
      std::unordered_set<KwargDataflowOutput<std::string>> correct =
          unordered_set_of(values(added.outputs));
      REQUIRE(result == correct);
    }

    KwargNodeAddedResult<std::string> added2 = g.add_node(
        /*inputs=*/
        {
            {
                "input_1",
                added_output_1,
            },
            {
                "input_2",
                added_output_3,
            },
        },
        /*output_slots=*/{
            "output_1",
        });

    KwargDataflowOutput<std::string> added2_output_1 =
        KwargDataflowOutput<std::string>{
            added2.outputs.at("output_1"),
        };

    {
      std::unordered_set<Node> result = g.query_nodes(node_query_all());
      std::unordered_set<Node> correct = {added.node, added2.node};
      REQUIRE(result == correct);
    }

    {
      std::unordered_set<KwargDataflowEdge<std::string>> result =
          g.query_edges(kwarg_dataflow_edge_query_all<std::string>());

      auto mk_edge =
          [](KwargDataflowOutput<std::string> const &src,
             Node const &dst_node,
             std::string const &dst_slot) -> KwargDataflowEdge<std::string> {
        return KwargDataflowEdge<std::string>{
            /*src=*/src,
            /*dst=*/
            KwargDataflowInput<std::string>{
                dst_node,
                dst_slot,
            },
        };
      };

      std::unordered_set<KwargDataflowEdge<std::string>> correct = {
          mk_edge(added_output_1, added2.node, "input_1"),
          mk_edge(added_output_3, added2.node, "input_2"),
      };

      REQUIRE(result == correct);
    }

    {
      std::unordered_set<KwargDataflowOutput<std::string>> result =
          g.query_outputs(kwarg_dataflow_output_query_all<std::string>());

      auto get_output_set = [](KwargNodeAddedResult<std::string> const &r) {
        return unordered_set_of(values(r.outputs));
      };

      std::unordered_set<KwargDataflowOutput<std::string>> correct =
          set_union(get_output_set(added), get_output_set(added2));

      REQUIRE(result == correct);
    }
  }
}
