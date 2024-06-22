// THIS FILE WAS AUTO-GENERATED BY proj. DO NOT MODIFY IT!
// If you would like to modify this datatype, instead modify
// lib/local-execution/include/local-execution/cost_details.struct.toml
/* proj-data
{
  "generated_from": "693db06746072111153062c0f087f4b6"
}
*/

#ifndef _FLEXFLOW_LIB_LOCAL_EXECUTION_INCLUDE_LOCAL_EXECUTION_COST_DETAILS_DTG_H
#define _FLEXFLOW_LIB_LOCAL_EXECUTION_INCLUDE_LOCAL_EXECUTION_COST_DETAILS_DTG_H

#include "fmt/format.h"
#include "nlohmann/json.hpp"
#include "rapidcheck.h"
#include <functional>
#include <ostream>
#include <tuple>

namespace FlexFlow {
struct CostDetails {
  CostDetails() = delete;
  CostDetails(float const &total_elapsed_time, size_t const &total_mem_usage);

  bool operator==(CostDetails const &) const;
  bool operator!=(CostDetails const &) const;
  bool operator<(CostDetails const &) const;
  bool operator>(CostDetails const &) const;
  bool operator<=(CostDetails const &) const;
  bool operator>=(CostDetails const &) const;
  float total_elapsed_time;
  size_t total_mem_usage;
};
} // namespace FlexFlow

namespace std {
template <>
struct hash<FlexFlow::CostDetails> {
  size_t operator()(FlexFlow::CostDetails const &) const;
};
} // namespace std

namespace nlohmann {
template <>
struct adl_serializer<FlexFlow::CostDetails> {
  static FlexFlow::CostDetails from_json(json const &);
  static void to_json(json &, FlexFlow::CostDetails const &);
};
} // namespace nlohmann

namespace rc {
template <>
struct Arbitrary<FlexFlow::CostDetails> {
  static Gen<FlexFlow::CostDetails> arbitrary();
};
} // namespace rc

namespace FlexFlow {
std::string format_as(CostDetails const &);
std::ostream &operator<<(std::ostream &, CostDetails const &);
} // namespace FlexFlow

#endif // _FLEXFLOW_LIB_LOCAL_EXECUTION_INCLUDE_LOCAL_EXECUTION_COST_DETAILS_DTG_H