#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_NUM_PTENSOR_PARALLEL_DIMS_T_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_NUM_PTENSOR_PARALLEL_DIMS_T_H

#include <functional>
#include <nlohmann/json.hpp>
#include <rapidcheck.h>
#include <string>
#include "utils/nonnegative_int/nonnegative_int.h"
#include "utils/positive_int/positive_int.h"

namespace FlexFlow {

struct num_ptensor_parallel_dims_t {
public:
  num_ptensor_parallel_dims_t() = delete;
  explicit num_ptensor_parallel_dims_t(int);
  explicit num_ptensor_parallel_dims_t(nonnegative_int);
  explicit num_ptensor_parallel_dims_t(positive_int);

  bool operator<(num_ptensor_parallel_dims_t const &other) const;
  bool operator==(num_ptensor_parallel_dims_t const &other) const;
  bool operator>(num_ptensor_parallel_dims_t const &other) const;
  bool operator<=(num_ptensor_parallel_dims_t const &other) const;
  bool operator!=(num_ptensor_parallel_dims_t const &other) const;
  bool operator>=(num_ptensor_parallel_dims_t const &other) const;

  int int_from_num_ptensor_parallel_dims() const;
  nonnegative_int nonnegative_int_from_num_ptensor_parallel_dims() const;
  positive_int positive_int_from_num_ptensor_parallel_dims() const;

private:
  int value;
private:
  void check_invariant() const;
};

std::ostream &operator<<(std::ostream &, num_ptensor_parallel_dims_t const &);
std::string format_as(num_ptensor_parallel_dims_t const &);

} // namespace FlexFlow

namespace nlohmann {

template <>
struct adl_serializer<::FlexFlow::num_ptensor_parallel_dims_t> {
  static ::FlexFlow::num_ptensor_parallel_dims_t from_json(json const &j);
  static void to_json(json &j, ::FlexFlow::num_ptensor_parallel_dims_t t);
};

} // namespace nlohmann

namespace rc {

template <>
struct Arbitrary<::FlexFlow::num_ptensor_parallel_dims_t> {
  static Gen<::FlexFlow::num_ptensor_parallel_dims_t> arbitrary();
};

} // namespace rc

namespace std {

template <>
struct hash<::FlexFlow::num_ptensor_parallel_dims_t> {
  size_t operator()(::FlexFlow::num_ptensor_parallel_dims_t const &) const noexcept;
};

} // namespace std

#endif
