#include "utils/graph/labelled_dataflow_graph/algorithms/find_isomorphism.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using NodeLabel = value_type<0>;
using ValueLabel = value_type<1>;

template
  std::optional<DataflowGraphIsomorphism> find_isomorphism(
      LabelledDataflowGraphView<NodeLabel, ValueLabel> const &,
      LabelledDataflowGraphView<NodeLabel, ValueLabel> const &);

} // namespace FlexFlow
