#include "substitutions/unlabelled/unlabelled_graph_pattern.h"
#include <doctest/doctest.h>
#include "utils/containers/require_only_key.h"
#include "utils/graph/instances/unordered_set_open_kwarg_dataflow_graph.h"
#include "utils/graph/open_kwarg_dataflow_graph/open_kwarg_dataflow_graph.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("is_singleton_pattern") {
    OpenKwargDataflowGraph<int, TensorSlotName> g =
        OpenKwargDataflowGraph<int, TensorSlotName>::create<
          UnorderedSetOpenKwargDataflowGraph<int, TensorSlotName>>();

    SUBCASE("0 nodes") {
      UnlabelledGraphPattern pattern = UnlabelledGraphPattern{g};

      CHECK_FALSE(is_singleton_pattern(pattern));
    }

    KwargNodeAddedResult n0_added = g.add_node({}, {TensorSlotName::OUTPUT});
    OpenKwargDataflowValue v0 = OpenKwargDataflowValue<int, TensorSlotName>{
      require_only_key(n0_added.outputs, TensorSlotName::OUTPUT),
    };

    SUBCASE("1 node") {
      UnlabelledGraphPattern pattern = UnlabelledGraphPattern{g};

      CHECK(is_singleton_pattern(pattern));
    }

    KwargNodeAddedResult n1_added = g.add_node({{TensorSlotName::INPUT, v0}}, {TensorSlotName::OUTPUT});
    OpenKwargDataflowValue<int, TensorSlotName> v1 = OpenKwargDataflowValue<int, TensorSlotName>{
      require_only_key(n1_added.outputs, TensorSlotName::OUTPUT),
    };

    SUBCASE("more than 1 node") {
      UnlabelledGraphPattern pattern = UnlabelledGraphPattern{g};

      CHECK_FALSE(is_singleton_pattern(pattern));
    }
  }
}
