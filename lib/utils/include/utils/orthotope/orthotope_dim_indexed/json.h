#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_ORTHOTOPE_ORTHOTOPE_DIM_INDEXED_JSON_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_ORTHOTOPE_ORTHOTOPE_DIM_INDEXED_JSON_H

#include "utils/orthotope/orthotope_dim_indexed/orthotope_dim_indexed.h"
#include "utils/orthotope/orthotope_dim_indexed/orthotope_dim_indexed_of.h"
#include <nlohmann/json.hpp>

namespace nlohmann {

template <typename T>
struct adl_serializer<::FlexFlow::OrthotopeDimIndexed<T>> {
  static ::FlexFlow::OrthotopeDimIndexed<T> from_json(json const &j) {
    return ::FlexFlow::orthotope_dim_indexed_of(j.get<std::vector<T>>());
  }

  static void to_json(json &j, ::FlexFlow::OrthotopeDimIndexed<T> const &d) {
    j = d.get_contents();
  }
};

} // namespace nlohmann

#endif
