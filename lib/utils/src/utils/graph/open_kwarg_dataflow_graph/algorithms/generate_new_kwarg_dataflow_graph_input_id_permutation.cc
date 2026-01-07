#include "utils/graph/open_kwarg_dataflow_graph/algorithms/generate_new_kwarg_dataflow_graph_input_id_permutation.h"
#include "utils/archetypes/ordered_value_type.h"

namespace FlexFlow {

using GraphInputName = ordered_value_type<0>;
using SlotName = ordered_value_type<1>;

template bidict<KwargDataflowGraphInput<GraphInputName>,
                KwargDataflowGraphInput<GraphInputName>>
    generate_new_kwarg_dataflow_graph_input_id_permutation(
        OpenKwargDataflowGraphView<GraphInputName, SlotName> const &,
        std::function<GraphInputName()> const &);

} // namespace FlexFlow
