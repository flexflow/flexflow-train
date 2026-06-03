#include "utils/graph/dataflow_graph/algorithms/dataflow_graph_as_dot.h"
#include "utils/containers/get_only.h"
#include "utils/graph/dataflow_graph/dataflow_graph.h"
#include "utils/graph/instances/unordered_set_dataflow_graph.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("dataflow_graph_as_dot") {
    DataflowGraph g = DataflowGraph::create<UnorderedSetDataflowGraph>();

    NodeAddedResult n1_added = g.add_node({}, 1_n);
    Node n1 = n1_added.node;
    DataflowOutput o1 = get_only(n1_added.outputs);

    NodeAddedResult n2_added = g.add_node({}, 1_n);
    Node n2 = n2_added.node;
    DataflowOutput o2 = get_only(n2_added.outputs);

    NodeAddedResult n3_added = g.add_node({}, 1_n);
    Node n3 = n3_added.node;
    DataflowOutput o3 = get_only(n3_added.outputs);

    NodeAddedResult n4_added = g.add_node({o1, o2, o3}, 1_n);
    Node n4 = n4_added.node;
    DataflowOutput o4 = get_only(n4_added.outputs);

    auto get_node_label = [&](Node n) -> std::string {
      if (n == n1) {
        return "n1";
      } else if (n == n2) {
        return "n2";
      } else if (n == n3) {
        return "n3";
      } else {
        ASSERT(n == n4);
        return "n4";
      }
    };

    auto get_input_label = [&](DataflowInput const &i) -> std::string {
      return fmt::format("{}_{}", get_node_label(i.node), i.idx);
    };

    auto get_output_label = [&](DataflowOutput const &o) -> std::string {
      return fmt::format("{}_{}", get_node_label(o.node), o.idx);
    };

    std::string result = dataflow_graph_as_dot(
        g, get_node_label, get_input_label, get_output_label);

    std::string correct = R"EXPECTED_OUTPUT(digraph taskgraph {
  node0 [label=<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0"><TR><TD COLSPAN="1">(no inputs)</TD></TR>
<TR><TD COLSPAN="1">n1</TD></TR>
<TR><TD PORT="o0" COLSPAN="1">n1_0</TD></TR></TABLE>>,shape=plaintext];
  node1 [label=<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0"><TR><TD COLSPAN="1">(no inputs)</TD></TR>
<TR><TD COLSPAN="1">n2</TD></TR>
<TR><TD PORT="o0" COLSPAN="1">n2_0</TD></TR></TABLE>>,shape=plaintext];
  node2 [label=<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0"><TR><TD COLSPAN="1">(no inputs)</TD></TR>
<TR><TD COLSPAN="1">n3</TD></TR>
<TR><TD PORT="o0" COLSPAN="1">n3_0</TD></TR></TABLE>>,shape=plaintext];
  node3 [label=<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0"><TR><TD PORT="i0" COLSPAN="1">n4_0</TD>
<TD PORT="i1" COLSPAN="1">n4_1</TD>
<TD PORT="i2" COLSPAN="1">n4_2</TD></TR>
<TR><TD COLSPAN="3">n4</TD></TR>
<TR><TD PORT="o0" COLSPAN="3">n4_0</TD></TR></TABLE>>,shape=plaintext];
  node1:o0 -> node3:i1;
  node2:o0 -> node3:i2;
  node0:o0 -> node3:i0;
})EXPECTED_OUTPUT";

    CHECK(result == correct);
  }
}
