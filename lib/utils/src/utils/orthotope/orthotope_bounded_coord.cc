#include "utils/orthotope/orthotope_bounded_coord.h"
#include "utils/containers/concat_vectors.h"
#include "utils/containers/get_last.h"
#include "utils/containers/get_only.h"
#include "utils/containers/product.h"
#include "utils/containers/require_same.h"
#include "utils/containers/slice.h"
#include "utils/containers/transform.h"
#include "utils/containers/zip_with_strict.h"
#include "utils/hash-utils.h"
#include "utils/hash/tuple.h"
#include "utils/int_ge_two/int_ge_two.h"
#include "utils/nonnegative_int/num_elements.h"
#include "utils/orthotope/orthotope.h"

namespace FlexFlow {

OrthotopeBoundedCoord::OrthotopeBoundedCoord(OrthotopeCoord const &c,
                                             Orthotope const &b)
    : coord(c), bounds(b) {
  ASSERT(orthotope_contains_coord(this->bounds, this->coord));
}

bool OrthotopeBoundedCoord::operator==(
    OrthotopeBoundedCoord const &other) const {
  return this->tie() == other.tie();
}

bool OrthotopeBoundedCoord::operator!=(
    OrthotopeBoundedCoord const &other) const {
  return this->tie() != other.tie();
}

bool OrthotopeBoundedCoord::operator<(
    OrthotopeBoundedCoord const &other) const {
  return this->tie() < other.tie();
}

bool OrthotopeBoundedCoord::operator>(
    OrthotopeBoundedCoord const &other) const {
  return this->tie() > other.tie();
}

bool OrthotopeBoundedCoord::operator<=(
    OrthotopeBoundedCoord const &other) const {
  return this->tie() <= other.tie();
}

bool OrthotopeBoundedCoord::operator>=(
    OrthotopeBoundedCoord const &other) const {
  return this->tie() >= other.tie();
}

OrthotopeCoord const &OrthotopeBoundedCoord::get_coord() const {
  return this->coord;
}

Orthotope const &OrthotopeBoundedCoord::get_bounds() const {
  return this->bounds;
}

std::tuple<OrthotopeCoord const &, Orthotope const &>
    OrthotopeBoundedCoord::tie() const {
  return std::tie(this->coord, this->bounds);
}

nlohmann::json format_as(OrthotopeBoundedCoord const &c) {
  nlohmann::json j = c;
  return j;
}

std::ostream &operator<<(std::ostream &s, OrthotopeBoundedCoord const &x) {
  return (s << fmt::to_string(x));
}

nonnegative_int
    orthotope_bounded_coord_num_dims(OrthotopeBoundedCoord const &c) {
  return require_same(num_elements(c.get_coord().raw),
                      num_elements(c.get_bounds().dims));
}

std::vector<BoundedComponent>
    components_of_orthotope_bounded_coord(OrthotopeBoundedCoord const &c) {
  return zip_with_strict(
      c.get_coord().raw,
      c.get_bounds().dims,
      [&](nonnegative_int component, positive_int bound) -> BoundedComponent {
        return BoundedComponent{
            /*component=*/component,
            /*bound=*/bound,
        };
      });
}

OrthotopeBoundedCoord lift_bounded_component(BoundedComponent const &c) {
  return OrthotopeBoundedCoord{
      /*coord=*/OrthotopeCoord{
          /*raw=*/{
              c.get_component(),
          },
      },
      /*bounds=*/
      Orthotope{
          /*dims=*/{
              c.get_bound(),
          },
      },
  };
}

OrthotopeBoundedCoord
    make_2d_orthotope_bounded_coord(BoundedComponent const &c1,
                                    BoundedComponent const &c2) {
  return OrthotopeBoundedCoord{
      /*coord=*/OrthotopeCoord{
          /*raw=*/{
              c1.get_component(),
              c2.get_component(),
          },
      },
      /*bounds=*/
      Orthotope{
          /*dims=*/{
              c1.get_bound(),
              c2.get_bound(),
          },
      },
  };
}

OrthotopeBoundedCoord
    make_3d_orthotope_bounded_coord(BoundedComponent const &c1,
                                    BoundedComponent const &c2,
                                    BoundedComponent const &c3) {
  return OrthotopeBoundedCoord{
      /*coord=*/OrthotopeCoord{
          /*raw=*/{
              c1.get_component(),
              c2.get_component(),
              c3.get_component(),
          },
      },
      /*bounds=*/
      Orthotope{
          /*dims=*/{
              c1.get_bound(),
              c2.get_bound(),
              c3.get_bound(),
          },
      },
  };
}

OrthotopeBoundedCoord
    make_4d_orthotope_bounded_coord(BoundedComponent const &c1,
                                    BoundedComponent const &c2,
                                    BoundedComponent const &c3,
                                    BoundedComponent const &c4) {
  return make_orthotope_bounded_coord_from_components(std::vector{
      c1,
      c2,
      c3,
      c4,
  });
}

OrthotopeBoundedCoord make_orthotope_bounded_coord_from_head_and_tail(
    BoundedComponent const &head, OrthotopeBoundedCoord const &tail) {
  std::vector<BoundedComponent> tail_components =
      components_of_orthotope_bounded_coord(tail);

  return make_orthotope_bounded_coord_from_components(
      concat_vectors(std::vector{head}, tail_components));
}

OrthotopeBoundedCoord make_orthotope_bounded_coord_from_components(
    std::vector<BoundedComponent> const &components) {
  return OrthotopeBoundedCoord{
      /*coord=*/OrthotopeCoord{
          /*raw=*/
          transform(components,
                    [](BoundedComponent const &c) -> nonnegative_int {
                      return c.get_component();
                    }),
      },
      /*bounds=*/
      Orthotope{
          /*dims=*/
          transform(components,
                    [](BoundedComponent const &c) -> positive_int {
                      return c.get_bound();
                    }),
      },
  };
}

OrthotopeBoundedCoord
    orthotope_bounded_coord_product(OrthotopeBoundedCoord const &c1,
                                    OrthotopeBoundedCoord const &c2) {
  return OrthotopeBoundedCoord{
      /*coord=*/OrthotopeCoord{
          concat_vectors(c1.get_coord().raw, c2.get_coord().raw),
      },
      /*bounds=*/
      Orthotope{
          concat_vectors(c1.get_bounds().dims, c2.get_bounds().dims),
      },
  };
}

OrthotopeBoundedCoord
    orthotope_bounded_coord_product(OrthotopeBoundedCoord const &c1,
                                    OrthotopeBoundedCoord const &c2,
                                    OrthotopeBoundedCoord const &c3) {
  return OrthotopeBoundedCoord{
      /*coord=*/OrthotopeCoord{
          concat_vectors(std::vector{
              c1.get_coord().raw, c2.get_coord().raw, c3.get_coord().raw}),
      },
      /*bounds=*/
      Orthotope{
          concat_vectors(std::vector{c1.get_bounds().dims,
                                     c2.get_bounds().dims,
                                     c3.get_bounds().dims}),
      },
  };
}

OrthotopeBoundedCoord orthotope_bounded_coord_product(
    std::vector<OrthotopeBoundedCoord> const &cs) {
  return OrthotopeBoundedCoord{
      /*coord=*/OrthotopeCoord{
          concat_vectors(transform(cs,
                                   [](OrthotopeBoundedCoord const &c)
                                       -> std::vector<nonnegative_int> {
                                     return c.get_coord().raw;
                                   })),
      },
      /*bounds=*/
      Orthotope{
          concat_vectors(transform(
              cs,
              [](OrthotopeBoundedCoord const &c) -> std::vector<positive_int> {
                return c.get_bounds().dims;
              })),
      },
  };
}

std::pair<BoundedComponent, OrthotopeBoundedCoord>
    orthotope_bounded_coord_slice_first(OrthotopeBoundedCoord const &coord) {
  std::vector<BoundedComponent> components =
      components_of_orthotope_bounded_coord(coord);

  BoundedComponent head = components.at(0);
  std::vector<BoundedComponent> tail = slice(components, 1, std::nullopt);

  return std::pair{
      head,
      make_orthotope_bounded_coord_from_components(tail),
  };
}

std::pair<OrthotopeBoundedCoord, BoundedComponent>
    orthotope_bounded_coord_slice_last(OrthotopeBoundedCoord const &coord) {
  std::vector<BoundedComponent> components =
      components_of_orthotope_bounded_coord(coord);

  BoundedComponent last = get_last(components);
  std::vector<BoundedComponent> leading = slice(components, 0, -1);

  return std::pair{
      make_orthotope_bounded_coord_from_components(leading),
      last,
  };
}

std::optional<BoundedComponent>
    flatten_orthotope_bounded_coord(OrthotopeBoundedCoord const &input) {
  nonnegative_int input_num_dims = orthotope_bounded_coord_num_dims(input);

  if (input_num_dims == 0) {
    return std::nullopt;
  }

  if (input_num_dims == 1) {
    return get_only(components_of_orthotope_bounded_coord(input));
  }

  std::pair<OrthotopeBoundedCoord, BoundedComponent> r =
      orthotope_bounded_coord_slice_last(input);
  OrthotopeBoundedCoord leading_components = r.first;
  BoundedComponent last_component = r.second;

  BoundedComponent flattened_leading =
      assert_unwrap(flatten_orthotope_bounded_coord(leading_components));

  return BoundedComponent{
      /*component=*/flattened_leading.get_component() *
              last_component.get_bound() +
          last_component.get_component(),
      /*bound=*/flattened_leading.get_bound() * last_component.get_bound(),
  };
}

} // namespace FlexFlow

namespace std {

size_t hash<::FlexFlow::OrthotopeBoundedCoord>::operator()(
    ::FlexFlow::OrthotopeBoundedCoord const &x) const {
  return ::FlexFlow::get_std_hash(x.tie());
}

} // namespace std

namespace nlohmann {

::FlexFlow::OrthotopeBoundedCoord
    adl_serializer<::FlexFlow::OrthotopeBoundedCoord>::from_json(
        json const &j) {

  return ::FlexFlow::OrthotopeBoundedCoord{
      /*coord=*/j.at("coord").template get<::FlexFlow::OrthotopeCoord>(),
      /*bounds=*/j.at("bounds").template get<::FlexFlow::Orthotope>(),
  };
}

void adl_serializer<::FlexFlow::OrthotopeBoundedCoord>::to_json(
    json &j, ::FlexFlow::OrthotopeBoundedCoord const &x) {
  j["__type"] = "OrthotopeBoundedCoord";
  j["coord"] = x.get_coord();
  j["bounds"] = x.get_bounds();
}

} // namespace nlohmann

namespace rc {

Gen<::FlexFlow::OrthotopeBoundedCoord>
    Arbitrary<::FlexFlow::OrthotopeBoundedCoord>::arbitrary() {
  using namespace FlexFlow;

  return gen::map(gen::arbitrary<std::vector<BoundedComponent>>(),
                  [](std::vector<BoundedComponent> const &components)
                      -> OrthotopeBoundedCoord {
                    return make_orthotope_bounded_coord_from_components(
                        components);
                  });
}

} // namespace rc
