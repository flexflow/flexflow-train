#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_ORTHOTOPE_ORTHOTOPE_BOUNDED_COORD_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_ORTHOTOPE_ORTHOTOPE_BOUNDED_COORD_H

#include "utils/fmt/json.h"
#include "utils/orthotope/bounded_component.h"
#include "utils/orthotope/orthotope.dtg.h"
#include "utils/orthotope/orthotope_coord.dtg.h"
#include <optional>

namespace FlexFlow {

struct OrthotopeBoundedCoord {
public:
  OrthotopeBoundedCoord() = delete;
  explicit OrthotopeBoundedCoord(OrthotopeCoord const &coord,
                                 Orthotope const &bounds);

  [[nodiscard]] bool operator==(OrthotopeBoundedCoord const &) const;
  [[nodiscard]] bool operator!=(OrthotopeBoundedCoord const &) const;

  [[nodiscard]] bool operator<(OrthotopeBoundedCoord const &) const;
  [[nodiscard]] bool operator>(OrthotopeBoundedCoord const &) const;
  [[nodiscard]] bool operator<=(OrthotopeBoundedCoord const &) const;
  [[nodiscard]] bool operator>=(OrthotopeBoundedCoord const &) const;

  [[nodiscard]] OrthotopeCoord const &get_coord() const;
  [[nodiscard]] Orthotope const &get_bounds() const;

private:
  OrthotopeCoord coord;
  Orthotope bounds;

private:
  [[nodiscard]] std::tuple<decltype(coord) const &, decltype(bounds) const &>
      tie() const;

  friend ::std::hash<::FlexFlow::OrthotopeBoundedCoord>;
};

nlohmann::json format_as(OrthotopeBoundedCoord const &);
std::ostream &operator<<(std::ostream &, OrthotopeBoundedCoord const &);

nonnegative_int orthotope_bounded_coord_num_dims(OrthotopeBoundedCoord const &);

std::vector<BoundedComponent>
    components_of_orthotope_bounded_coord(OrthotopeBoundedCoord const &);

OrthotopeBoundedCoord lift_bounded_component(BoundedComponent const &);

OrthotopeBoundedCoord make_2d_orthotope_bounded_coord(BoundedComponent const &,
                                                      BoundedComponent const &);

OrthotopeBoundedCoord make_3d_orthotope_bounded_coord(BoundedComponent const &,
                                                      BoundedComponent const &,
                                                      BoundedComponent const &);

OrthotopeBoundedCoord make_4d_orthotope_bounded_coord(BoundedComponent const &,
                                                      BoundedComponent const &,
                                                      BoundedComponent const &,
                                                      BoundedComponent const &);

OrthotopeBoundedCoord make_orthotope_bounded_coord_from_head_and_tail(
    BoundedComponent const &head, OrthotopeBoundedCoord const &tail);

OrthotopeBoundedCoord make_orthotope_bounded_coord_from_components(
    std::vector<BoundedComponent> const &);

OrthotopeBoundedCoord
    orthotope_bounded_coord_product(OrthotopeBoundedCoord const &,
                                    OrthotopeBoundedCoord const &);

OrthotopeBoundedCoord
    orthotope_bounded_coord_product(OrthotopeBoundedCoord const &,
                                    OrthotopeBoundedCoord const &,
                                    OrthotopeBoundedCoord const &);

OrthotopeBoundedCoord
    orthotope_bounded_coord_product(std::vector<OrthotopeBoundedCoord> const &);

std::pair<BoundedComponent, OrthotopeBoundedCoord>
    orthotope_bounded_coord_slice_first(OrthotopeBoundedCoord const &);

std::pair<OrthotopeBoundedCoord, BoundedComponent>
    orthotope_bounded_coord_slice_last(OrthotopeBoundedCoord const &);

std::optional<BoundedComponent>
    flatten_orthotope_bounded_coord(OrthotopeBoundedCoord const &);

} // namespace FlexFlow

namespace std {

template <>
struct hash<::FlexFlow::OrthotopeBoundedCoord> {
  size_t operator()(::FlexFlow::OrthotopeBoundedCoord const &) const;
};

} // namespace std

namespace nlohmann {

template <>
struct adl_serializer<::FlexFlow::OrthotopeBoundedCoord> {
  static ::FlexFlow::OrthotopeBoundedCoord from_json(json const &);
  static void to_json(json &, ::FlexFlow::OrthotopeBoundedCoord const &);
};

} // namespace nlohmann

namespace rc {

template <>
struct Arbitrary<::FlexFlow::OrthotopeBoundedCoord> {
  static Gen<::FlexFlow::OrthotopeBoundedCoord> arbitrary();
};

} // namespace rc

#endif
