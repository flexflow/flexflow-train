#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_OPEN_DATAFLOW_GRAPH_JSON_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_OPEN_DATAFLOW_GRAPH_JSON_H

#include "utils/graph/instances/unordered_set_labelled_open_dataflow_graph.h"
#include "utils/graph/labelled_open_dataflow_graph/algorithms/from_labelled_open_dataflow_graph_data.h"
#include "utils/graph/labelled_open_dataflow_graph/algorithms/get_graph_data.h"
#include "utils/graph/labelled_open_dataflow_graph/algorithms/labelled_open_dataflow_graph_data.dtg.h"
#include "utils/graph/labelled_open_dataflow_graph/labelled_open_dataflow_graph.h"
#include "utils/json/check_is_json_deserializable.h"
#include "utils/json/check_is_json_serializable.h"
#include <nlohmann/json.hpp>

namespace nlohmann {

template <typename NodeLabel, typename ValueLabel>
struct adl_serializer<
    ::FlexFlow::LabelledOpenDataflowGraph<NodeLabel, ValueLabel>> {
  static ::FlexFlow::LabelledOpenDataflowGraph<NodeLabel, ValueLabel>
      from_json(json const &j) {
    CHECK_IS_JSON_DESERIALIZABLE(
        ::FlexFlow::LabelledOpenDataflowGraphData<NodeLabel, ValueLabel>);

    auto data = j.template get<
        ::FlexFlow::LabelledOpenDataflowGraphData<NodeLabel, ValueLabel>>();
    ::FlexFlow::LabelledOpenDataflowGraphView<NodeLabel, ValueLabel> view =
        ::FlexFlow::from_labelled_open_dataflow_graph_data(data);
    return ::FlexFlow::LabelledOpenDataflowGraph<NodeLabel, ValueLabel>::
        template create_copy_of<
            ::FlexFlow::UnorderedSetLabelledOpenDataflowGraph<NodeLabel,
                                                              ValueLabel>>(
            view);
  }

  static void to_json(
      json &j,
      ::FlexFlow::LabelledOpenDataflowGraph<NodeLabel, ValueLabel> const &g) {
    CHECK_IS_JSON_SERIALIZABLE(
        ::FlexFlow::LabelledOpenDataflowGraphData<NodeLabel, ValueLabel>);

    j = ::FlexFlow::get_graph_data(g);
  }
};

} // namespace nlohmann

#endif
