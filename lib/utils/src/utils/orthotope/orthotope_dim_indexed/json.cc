#include "utils/orthotope/orthotope_dim_indexed/json.h"

namespace nlohmann {

template 
  struct adl_serializer<::FlexFlow::OrthotopeDimIndexed<int>>;

} // namespace nlohmann
