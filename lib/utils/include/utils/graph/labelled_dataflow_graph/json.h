#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_DATAFLOW_GRAPH_JSON_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_DATAFLOW_GRAPH_JSON_H

#include "utils/graph/instances/unordered_set_labelled_open_dataflow_graph.h"
#include "utils/graph/labelled_dataflow_graph/algorithms/view_as_labelled_open_dataflow_graph.h"
#include "utils/graph/labelled_dataflow_graph/labelled_dataflow_graph.h"
#include "utils/graph/labelled_open_dataflow_graph/algorithms/from_labelled_open_dataflow_graph_data.h"
#include "utils/graph/labelled_open_dataflow_graph/algorithms/get_graph_data.h"
#include "utils/graph/labelled_open_dataflow_graph/algorithms/labelled_open_dataflow_graph_data.dtg.h"
#include "utils/json/check_is_json_deserializable.h"
#include "utils/json/check_is_json_serializable.h"
#include <nlohmann/json.hpp>

namespace nlohmann {

template <typename NodeLabel, typename OutputLabel>
struct adl_serializer<
    ::FlexFlow::LabelledDataflowGraph<NodeLabel, OutputLabel>> {
  static ::FlexFlow::LabelledDataflowGraph<NodeLabel, OutputLabel>
      from_json(json const &j) {
    CHECK_IS_JSON_DESERIALIZABLE(
        ::FlexFlow::LabelledOpenDataflowGraphData<NodeLabel, OutputLabel>);

    auto data = j.template get<
        ::FlexFlow::LabelledOpenDataflowGraphData<NodeLabel, OutputLabel>>();
    ::FlexFlow::LabelledOpenDataflowGraphView<NodeLabel, OutputLabel>
        open_view = ::FlexFlow::from_labelled_open_dataflow_graph_data(data);
    return ::FlexFlow::LabelledDataflowGraph<NodeLabel, OutputLabel>::
        template create_copy_of<
            ::FlexFlow::UnorderedSetLabelledOpenDataflowGraph<NodeLabel,
                                                              OutputLabel>>(
            open_view);
  }

  static void to_json(
      json &j,
      ::FlexFlow::LabelledDataflowGraph<NodeLabel, OutputLabel> const &g) {
    CHECK_IS_JSON_SERIALIZABLE(
        ::FlexFlow::LabelledOpenDataflowGraphData<NodeLabel, OutputLabel>);

    ::FlexFlow::LabelledOpenDataflowGraphView<NodeLabel, OutputLabel>
        open_view = ::FlexFlow::view_as_labelled_open_dataflow_graph(g);
    j = ::FlexFlow::get_graph_data(open_view);
  }
};

} // namespace nlohmann

#endif
