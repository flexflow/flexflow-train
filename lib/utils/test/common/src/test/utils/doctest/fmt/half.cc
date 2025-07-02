#include "test/utils/doctest/fmt/half.h"

namespace doctest {

String StringMaker<::half>::convert(::half const &h) {
  return toString(static_cast<float>(h));
}

}
