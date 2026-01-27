#include "utils/graph/open_kwarg_dataflow_graph/algorithms/open_kwarg_dataflow_graph_isomorphism.h"
#include "utils/archetypes/ordered_value_type.h"

namespace FlexFlow {

using GraphInputName = ordered_value_type<0>;
using SlotName = ordered_value_type<1>;

template OpenKwargDataflowValue<GraphInputName, SlotName>
    isomorphism_map_r_open_kwarg_dataflow_value_from_l(
        OpenKwargDataflowGraphIsomorphism<GraphInputName> const &,
        OpenKwargDataflowValue<GraphInputName, SlotName> const &);

template OpenKwargDataflowValue<GraphInputName, SlotName>
    isomorphism_map_l_open_kwarg_dataflow_value_from_r(
        OpenKwargDataflowGraphIsomorphism<GraphInputName> const &,
        OpenKwargDataflowValue<GraphInputName, SlotName> const &);

template KwargDataflowOutput<SlotName>
    isomorphism_map_r_kwarg_dataflow_output_from_l(
        OpenKwargDataflowGraphIsomorphism<GraphInputName> const &,
        KwargDataflowOutput<SlotName> const &);

template KwargDataflowOutput<SlotName>
    isomorphism_map_l_kwarg_dataflow_output_from_r(
        OpenKwargDataflowGraphIsomorphism<GraphInputName> const &,
        KwargDataflowOutput<SlotName> const &);

} // namespace FlexFlow
