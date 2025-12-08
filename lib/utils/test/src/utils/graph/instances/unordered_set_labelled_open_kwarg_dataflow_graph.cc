#include <doctest/doctest.h>
#include "utils/containers/flatmap.h"
#include "utils/containers/values.h"
#include "utils/graph/instances/unordered_set_labelled_open_kwarg_dataflow_graph.h"
#include "utils/graph/labelled_open_kwarg_dataflow_graph/labelled_open_kwarg_dataflow_graph.h"
#include "utils/graph/node/node_query.h"
#include "utils/graph/open_kwarg_dataflow_graph/open_kwarg_dataflow_edge_query.h"
#include "utils/graph/kwarg_dataflow_graph/kwarg_dataflow_output_query.h"
#include "utils/singular_or_variadic.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("UnorderedSetLabelledOpenKwargDataflowGraph") {
    LabelledOpenKwargDataflowGraph<int, bool, std::string, std::string>
      g = LabelledOpenKwargDataflowGraph<int, bool, std::string, std::string>::
        create<UnorderedSetLabelledOpenKwargDataflowGraph<int, bool, std::string, std::string>>();

    {
      std::unordered_set<Node> result = g.query_nodes(node_query_all());
      std::unordered_set<Node> correct = {};
      REQUIRE(result == correct);
    }

    {
      std::unordered_set<OpenKwargDataflowEdge<std::string, std::string>> result =
          g.query_edges(open_kwarg_dataflow_edge_query_all<std::string, std::string>());
      std::unordered_set<OpenKwargDataflowEdge<std::string, std::string>> correct = {};
      REQUIRE(result == correct);
    }

    {
      std::unordered_set<KwargDataflowOutput<std::string>> result =
          g.query_outputs(kwarg_dataflow_output_query_all<std::string>());
      std::unordered_set<KwargDataflowOutput<std::string>> correct = {};
      REQUIRE(result == correct);
    }

    KwargNodeAddedResult<std::string> added = g.add_node(
      /*node_label=*/5,
      /*inputs=*/{},
      /*output_labels=*/{
        {
          "output_1",
          SingularOrVariadic{true}
        },
        {
          "output_2",
          SingularOrVariadic{
            std::vector{true, false, true},
          },
        },
      });

    KwargDataflowOutput<std::string> added_output_1 =
        added.outputs.at("output_1").require_singular();

    std::vector<KwargDataflowOutput<std::string>> added_output_2s = 
      added.outputs.at("output_2").require_variadic();
      
    {
      std::unordered_set<Node> result = g.query_nodes(node_query_all());
      std::unordered_set<Node> correct = {added.node};
      REQUIRE(result == correct);
    }

    {
      std::unordered_set<OpenKwargDataflowEdge<std::string, std::string>> result =
          g.query_edges(open_kwarg_dataflow_edge_query_all<std::string, std::string>());
      std::unordered_set<OpenKwargDataflowEdge<std::string, std::string>> correct = {};
      REQUIRE(result == correct);
    }

    {
      std::unordered_set<KwargDataflowOutput<std::string>> result =
          g.query_outputs(kwarg_dataflow_output_query_all<std::string>());
      std::unordered_set<KwargDataflowOutput<std::string>> correct =
          flatmap(unordered_set_of(values(added.outputs)),
                  [](SingularOrVariadic<KwargDataflowOutput<std::string>> const &s_or_v) {
                    return unordered_set_of(singular_or_variadic_values(s_or_v));
                  });
      REQUIRE(result == correct);
    }

    KwargDataflowGraphInput<std::string> input = 
      g.add_input("external_input", false);

    auto mk_open_val = [](auto const &v) -> OpenKwargDataflowValue<std::string, std::string> {
      return OpenKwargDataflowValue<std::string, std::string>{v};
    };

    KwargNodeAddedResult<std::string> added2 = g.add_node(
      /*node_label=*/5,
      /*inputs=*/{
        {
          "input_1",
          SingularOrVariadic{
            std::vector{
              mk_open_val(added_output_1),
              mk_open_val(added_output_2s.at(1)),
              mk_open_val(input),
              mk_open_val(added_output_2s.at(1)),
              mk_open_val(added_output_2s.at(0)),
            }
          },
        },
        {
          "input_2",
          SingularOrVariadic{
            mk_open_val(added_output_2s.at(1)),
          },
        },
      },
      /*output_labels=*/{
        {
          "output_1",
          SingularOrVariadic{
            true,
          },
        }
      });

    OpenKwargDataflowValue<std::string, std::string> added2_output_1 =
      OpenKwargDataflowValue<std::string, std::string>{
        added2.outputs.at("output_1").require_singular(),
      };

    {
      std::unordered_set<Node> result = g.query_nodes(node_query_all());
      std::unordered_set<Node> correct = {added.node, added2.node};
      REQUIRE(result == correct);
    }

    {
      std::unordered_set<OpenKwargDataflowEdge<std::string, std::string>> result =
          g.query_edges(open_kwarg_dataflow_edge_query_all<std::string, std::string>());

      auto internal_edge = [](KwargDataflowOutput<std::string> const &src,
                              Node const &dst_node,
                              std::string const &dst_slot) {
        return OpenKwargDataflowEdge<std::string, std::string>{
          KwargDataflowEdge{
            /*src=*/src,
            /*dst=*/KwargDataflowInput<std::string>{
              dst_node,
              dst_slot,
            },
          },
        };
      };

      auto external_edge = [](KwargDataflowGraphInput<std::string> const &src,
                              Node const &dst_node,
                              std::string const &dst_slot) {
        return OpenKwargDataflowEdge<std::string, std::string>{
          KwargDataflowInputEdge<std::string, std::string>{
            /*src=*/src,
            /*dst=*/KwargDataflowInput<std::string>{
              dst_node,
              dst_slot,
            },
          },
        };
      };

      std::unordered_set<OpenKwargDataflowEdge<std::string, std::string>> correct = {
        internal_edge(added_output_1, added2.node, "input_1"),
        internal_edge(added_output_2s.at(1), added2.node, "input_1"),
        external_edge(input, added2.node, "input_1"),
        internal_edge(added_output_2s.at(1), added2.node, "input_1"),
        internal_edge(added_output_2s.at(0), added2.node, "input_1"),
        internal_edge(added_output_2s.at(1), added2.node, "input_2"),
      };

      REQUIRE(result == correct);
    }

    {
      std::unordered_set<KwargDataflowOutput<std::string>> result =
          g.query_outputs(kwarg_dataflow_output_query_all<std::string>());

      auto get_output_set = [](KwargNodeAddedResult<std::string> const &r) {
        return flatmap(unordered_set_of(values(r.outputs)),
                  [](SingularOrVariadic<KwargDataflowOutput<std::string>> const &s_or_v) {
                    return unordered_set_of(singular_or_variadic_values(s_or_v));
                  });
      };

      std::unordered_set<KwargDataflowOutput<std::string>> correct =
        set_union(get_output_set(added), get_output_set(added2));

      REQUIRE(result == correct);
    }
  }
}
