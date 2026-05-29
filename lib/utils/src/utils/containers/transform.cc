#include "utils/containers/transform.h"
#include "utils/archetypes/value_type.h"
#include "utils/archetypes/ordered_value_type.h"

namespace FlexFlow {

using In = value_type<0>;
using Out = value_type<1>;
using F = std::function<Out(In const &)>;

template std::vector<Out> transform(std::vector<In> const &, F const &);

template std::unordered_set<Out> transform(std::unordered_set<In> const &, F const &);

template std::unordered_multiset<Out> transform(std::unordered_multiset<In> const &, F const &);

using In2 = ordered_value_type<0>;
using Out2 = ordered_value_type<1>;
using F2 = std::function<Out2(In2 const &)>;

template std::set<Out2> transform(std::set<In2> const &, F2 const &);

template std::multiset<Out2> transform(std::multiset<In2> const &v, F2 const &f);

using F3 = std::function<char(char)>;

template std::string transform(std::string const &, F3 const &);

using K = value_type<3>;
using V = value_type<4>;
using K2 = value_type<5>;
using V2 = value_type<6>;

template std::unordered_map<K2, V2> transform(std::unordered_map<K, V> const &,
                                              std::function<std::pair<K2, V2>(K const &, V const &)> const &);

using K3 = ordered_value_type<3>;
using V3 = value_type<4>;
using K4 = ordered_value_type<5>;
using V4 = value_type<6>;

template std::map<K4, V4> transform(std::map<K3, V3> const &,
                                    std::function<std::pair<K4, V4>(K3 const &, V3 const &)> const &);

template std::optional<Out> transform(std::optional<In> const &o,
                                      std::function<Out(In const &)> const &);
} // namespace FlexFlow
