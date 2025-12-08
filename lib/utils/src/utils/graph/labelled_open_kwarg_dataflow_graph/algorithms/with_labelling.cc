#include "utils/graph/labelled_open_kwarg_dataflow_graph/algorithms/with_labelling.h"
#include "utils/archetypes/value_type.h"
#include "utils/archetypes/ordered_value_type.h"

namespace FlexFlow {

template struct OpenKwargDataflowGraphLabellingWrapper<
  value_type<0>,
  value_type<1>,
  ordered_value_type<2>,
  ordered_value_type<3>>;

template struct OpenKwargDataflowGraphLabellingWrapper<
  ordered_value_type<0>,
  ordered_value_type<0>,
  ordered_value_type<0>,
  ordered_value_type<0>>;

using NodeLabel = value_type<0>;
using ValueLabel = value_type<1>;
using GraphInputName = ordered_value_type<2>;
using SlotName = ordered_value_type<3>;

template 
  LabelledOpenKwargDataflowGraphView<NodeLabel, ValueLabel, GraphInputName, SlotName> with_labelling(
      OpenKwargDataflowGraphView<GraphInputName, SlotName> const &, 
      std::unordered_map<Node, NodeLabel> const &,
      std::unordered_map<OpenKwargDataflowValue<GraphInputName, SlotName>, ValueLabel> const &);

} // namespace FlexFlow
