#ifndef _FLEXFLOW_LIB_UTILS_JSON_EXTRA_INCLUDE_UTILS_JSON_EXTRA_VISITABLE_SERIALIZER_INSTANCE_H
#define _FLEXFLOW_LIB_UTILS_JSON_EXTRA_INCLUDE_UTILS_JSON_EXTRA_VISITABLE_SERIALIZER_INSTANCE_H

namespace nlohmann {

template <typename T>
struct adl_serializer<
    T,
    typename std::enable_if<::FlexFlow::conjunction<
        ::FlexFlow::is_visitable<T>,
        ::FlexFlow::elements_satisfy<::FlexFlow::is_json_serializable, T>,
        std::is_default_constructible<T>>::value>::type> {
  static void to_json(json &j, T const &t) {
    ::FlexFlow::visit_json_serialize(j, t);
  }

  static void from_json(json const &j, T &t) {
    ::FlexFlow::visit_json_deserialize(j, t);
  }
};

} // namespace nlohmann

#endif