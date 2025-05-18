#include "kernels/array_shape.h"
#include "kernels/legion_ordered/slice.h"
#include "op-attrs/ff_ordered/ff_ordered_of.h"
#include "op-attrs/ff_ordered/slice.h"
#include "utils/containers/cartesian_product.h"
#include "utils/containers/product.h"
#include "utils/containers/reversed.h"
#include "utils/containers/transform.h"
#include "utils/containers/unordered_set_of.h"
#include "utils/containers/vector_of.h"
#include "utils/hash/tuple.h"
#include "utils/hash/vector.h"
#include "utils/nonnegative_int/num_elements.h"

namespace FlexFlow {

ArrayShape::ArrayShape(LegionOrdered<nonnegative_int> const &input_dims)
    : dims(input_dims) {}

nonnegative_int ArrayShape::get_volume() const {
  return this->num_elements();
}

nonnegative_int ArrayShape::num_dims() const {
  return ::FlexFlow::num_elements(this->dims);
}

nonnegative_int ArrayShape::get_dim() const {
  return this->num_dims();
}

nonnegative_int ArrayShape::num_elements() const {
  if (dims.size() == 0) {
    return 0_n;
  }
  return product(this->dims);
}

nonnegative_int ArrayShape::operator[](legion_dim_t idx) const {
  return dims.at(idx);
}

nonnegative_int ArrayShape::at(legion_dim_t idx) const {
  return dims.at(idx);
}

nonnegative_int ArrayShape::at(ff_dim_t idx) const {
  return dims.at(legion_dim_from_ff_dim(idx, this->num_dims()));
}

bool ArrayShape::operator==(ArrayShape const &other) const {
  return this->tie() == other.tie();
}

bool ArrayShape::operator!=(ArrayShape const &other) const {
  return this->tie() != other.tie();
}

ArrayShape
    ArrayShape::sub_shape(ff_dim_t const &start,
                          std::optional<ff_dim_t> const &maybe_end) const {
  FFOrdered<nonnegative_int> ff_ordered_dims =
      ff_ordered_from_legion_ordered(this->dims);
  FFOrdered<nonnegative_int> sliced = slice(ff_ordered_dims, start, maybe_end);
  return ArrayShape{legion_ordered_from_ff_ordered(sliced)};
}

ArrayShape
    ArrayShape::sub_shape(legion_dim_t const &start,
                          std::optional<legion_dim_t> const &maybe_end) const {
  return ArrayShape{slice(this->dims, start, maybe_end)};
}

std::optional<nonnegative_int> ArrayShape::at_maybe(legion_dim_t index) const {
  if (index.value < dims.size()) {
    return dims.at(index);
  } else {
    return std::nullopt;
  }
}

std::optional<nonnegative_int> ArrayShape::at_maybe(ff_dim_t index) const {
  return this->at_maybe(legion_dim_from_ff_dim(index, this->num_dims()));
}

std::tuple<LegionOrdered<nonnegative_int> const &> ArrayShape::tie() const {
  return std::tie(this->dims);
}

std::string format_as(ArrayShape const &x) {
  std::ostringstream oss;
  oss << "<ArrayShape";
  oss << " dims=" << x.dims;
  oss << ">";
  return oss.str();
}

std::ostream &operator<<(std::ostream &s, ArrayShape const &x) {
  return (s << fmt::to_string(x));
}

nonnegative_int get_volume(ArrayShape const &shape) {
  return shape.get_volume();
}

ArrayShape array_shape_from_tensor_shape(TensorShape const &tensor_shape) {
  return ArrayShape{
      legion_ordered_from_ff_ordered(tensor_shape.dims.ff_ordered)};
}

TensorShape get_tensor_shape(ArrayShape const &shape, DataType dtype) {
  return TensorShape{TensorDims{ff_ordered_from_legion_ordered(shape.dims)},
                     dtype};
}

std::unordered_set<ArrayCoord> get_array_coord_set(ArrayShape const &shape) {
  std::vector<std::vector<nonnegative_int>> per_dim_ranges =
      transform(vector_of(ff_ordered_from_legion_ordered(shape.dims)),
                [](nonnegative_int dim_size) -> std::vector<nonnegative_int> {
                  return nonnegative_range(dim_size);
                });

  std::unordered_set<std::vector<nonnegative_int>> raw_points =
      unordered_set_of(cartesian_product(per_dim_ranges));

  return transform(raw_points,
                   [](std::vector<nonnegative_int> const &raw_point) {
                     return ArrayCoord{ff_ordered_of(raw_point)};
                   });
}

} // namespace FlexFlow

namespace std {

using namespace FlexFlow;

size_t hash<ArrayShape>::operator()(ArrayShape const &s) const {
  return get_std_hash(s.tie());
}

} // namespace std
