#include "kernels/array_shape.h"
#include "utils/containers/product.h"
#include "utils/containers/reversed.h"
#include "utils/containers/vector_of.h"
#include "utils/nonnegative_int/num_elements.h"

namespace FlexFlow {

static LegionOrdered<nonnegative_int>
    legion_dims_from_ff_dims(FFOrdered<nonnegative_int> const &ff_ordered) {
  return LegionOrdered<nonnegative_int>{reversed(vector_of(ff_ordered))};
}

ArrayShape::ArrayShape(nonnegative_int const *_dims, nonnegative_int num_dims)
    : dims(_dims, _dims + num_dims.unwrap_nonnegative()) {}

ArrayShape::ArrayShape(TensorShape const &shape)
    : dims(legion_dims_from_ff_dims(shape.dims.ff_ordered)) {}

ArrayShape::ArrayShape(std::vector<nonnegative_int> const &input_dims)
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

ArrayShape ArrayShape::sub_shape(
    std::optional<std::variant<ff_dim_t, legion_dim_t>> start,
    std::optional<std::variant<ff_dim_t, legion_dim_t>> end) const {

  nonnegative_int num_dims = this->num_dims();

  auto to_legion_index = [num_dims](auto arg) -> nonnegative_int {
    using T = std::decay_t<decltype(arg)>;
    if constexpr (std::is_same_v<T, ff_dim_t>) {
      return legion_dim_from_ff_dim(arg, num_dims).value;
    } else {
      return arg.value;
    }
  };

  nonnegative_int start_idx =
      (start.has_value()) ? std::visit(to_legion_index, start.value()) : 0_n;

  nonnegative_int end_idx =
      (end.has_value()) ? std::visit(to_legion_index, end.value()) : num_dims;

  if (start_idx > num_dims || end_idx > num_dims || start_idx > end_idx) {
    throw mk_runtime_error(fmt::format(
        "Invalid sub_shape range: start={}, end={}", start_idx, end_idx));
  }

  return ArrayShape(&this->dims[legion_dim_t{start_idx}], end_idx - start_idx);
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

nonnegative_int get_volume(ArrayShape const &shape) {
  return shape.get_volume();
}

TensorShape get_tensor_shape(ArrayShape const &shape, DataType dtype) {
  return TensorShape{TensorDims{ff_ordered_from_legion_ordered(shape.dims)},
                     dtype};
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

} // namespace FlexFlow
