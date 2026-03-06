#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_KWARG_DATAFLOW_GRAPH_JSON_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_KWARG_DATAFLOW_GRAPH_JSON_H

#include "utils/graph/instances/unordered_set_kwarg_dataflow_graph.h"
#include "utils/graph/kwarg_dataflow_graph/algorithms/view_as_open_kwarg_dataflow_graph.h"
#include "utils/graph/kwarg_dataflow_graph/kwarg_dataflow_graph.h"
#include "utils/graph/open_kwarg_dataflow_graph/algorithms/get_open_kwarg_dataflow_graph_data.h"
#include "utils/graph/open_kwarg_dataflow_graph/algorithms/open_kwarg_dataflow_graph_data.dtg.h"
#include "utils/graph/open_kwarg_dataflow_graph/algorithms/view_from_open_kwarg_dataflow_graph_data.h"
#include "utils/json/check_is_json_deserializable.h"
#include "utils/json/check_is_json_serializable.h"
#include "utils/json/monostate.h"
#include <nlohmann/json.hpp>
#include <variant>

namespace nlohmann {

template <typename SlotName>
struct adl_serializer<::FlexFlow::KwargDataflowGraph<SlotName>> {
  static ::FlexFlow::KwargDataflowGraph<SlotName> from_json(json const &j) {
    CHECK_IS_JSON_DESERIALIZABLE(
        ::FlexFlow::OpenKwargDataflowGraphData<std::monostate, SlotName>);

    auto data = j.template get<
        ::FlexFlow::OpenKwargDataflowGraphData<std::monostate, SlotName>>();
    ::FlexFlow::OpenKwargDataflowGraphView<std::monostate, SlotName> view =
        ::FlexFlow::view_from_open_kwarg_dataflow_graph_data(data);
    return ::FlexFlow::KwargDataflowGraph<SlotName>::template create_copy_of<
        ::FlexFlow::UnorderedSetKwargDataflowGraph<SlotName>>(view);
  }

  static void to_json(json &j,
                      ::FlexFlow::KwargDataflowGraph<SlotName> const &g) {
    CHECK_IS_JSON_SERIALIZABLE(
        ::FlexFlow::OpenKwargDataflowGraphData<std::monostate, SlotName>);

    ::FlexFlow::OpenKwargDataflowGraphView<std::monostate, SlotName> open_view =
        ::FlexFlow::view_as_open_kwarg_dataflow_graph<std::monostate>(g);
    j = ::FlexFlow::get_open_kwarg_dataflow_graph_data(open_view);
  }
};

} // namespace nlohmann

#endif
