#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_OPEN_KWARG_DATAFLOW_GRAPH_JSON_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_OPEN_KWARG_DATAFLOW_GRAPH_JSON_H

#include "utils/graph/instances/unordered_set_labelled_open_kwarg_dataflow_graph.h"
#include "utils/graph/labelled_open_kwarg_dataflow_graph/algorithms/get_labelled_open_kwarg_dataflow_graph_data.h"
#include "utils/graph/labelled_open_kwarg_dataflow_graph/algorithms/labelled_open_kwarg_dataflow_graph_data.dtg.h"
#include "utils/graph/labelled_open_kwarg_dataflow_graph/algorithms/view_from_labelled_open_kwarg_dataflow_graph_data.h"
#include "utils/graph/labelled_open_kwarg_dataflow_graph/labelled_open_kwarg_dataflow_graph.h"
#include "utils/json/check_is_json_deserializable.h"
#include "utils/json/check_is_json_serializable.h"
#include <nlohmann/json.hpp>

namespace nlohmann {

template <typename NodeLabel,
          typename ValueLabel,
          typename GraphInputName,
          typename SlotName>
struct adl_serializer<::FlexFlow::LabelledOpenKwargDataflowGraph<NodeLabel,
                                                                 ValueLabel,
                                                                 GraphInputName,
                                                                 SlotName>> {
  static ::FlexFlow::LabelledOpenKwargDataflowGraph<NodeLabel,
                                                    ValueLabel,
                                                    GraphInputName,
                                                    SlotName>
      from_json(json const &j) {
    CHECK_IS_JSON_DESERIALIZABLE(
        ::FlexFlow::LabelledOpenKwargDataflowGraphData<NodeLabel,
                                                       ValueLabel,
                                                       GraphInputName,
                                                       SlotName>);

    auto data = j.template get<
        ::FlexFlow::LabelledOpenKwargDataflowGraphData<NodeLabel,
                                                       ValueLabel,
                                                       GraphInputName,
                                                       SlotName>>();
    ::FlexFlow::LabelledOpenKwargDataflowGraphView<NodeLabel,
                                                   ValueLabel,
                                                   GraphInputName,
                                                   SlotName>
        view =
            ::FlexFlow::view_from_labelled_open_kwarg_dataflow_graph_data(data);
    return ::FlexFlow::LabelledOpenKwargDataflowGraph<NodeLabel,
                                                      ValueLabel,
                                                      GraphInputName,
                                                      SlotName>::
        template create_copy_of<
            ::FlexFlow::UnorderedSetLabelledOpenKwargDataflowGraph<
                NodeLabel,
                ValueLabel,
                GraphInputName,
                SlotName>>(view);
  }

  static void
      to_json(json &j,
              ::FlexFlow::LabelledOpenKwargDataflowGraph<NodeLabel,
                                                         ValueLabel,
                                                         GraphInputName,
                                                         SlotName> const &g) {
    CHECK_IS_JSON_SERIALIZABLE(
        ::FlexFlow::LabelledOpenKwargDataflowGraphData<NodeLabel,
                                                       ValueLabel,
                                                       GraphInputName,
                                                       SlotName>);

    j = ::FlexFlow::get_labelled_open_kwarg_dataflow_graph_data(g);
  }
};

} // namespace nlohmann

#endif
