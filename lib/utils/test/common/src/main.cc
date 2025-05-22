#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest/doctest.h>

#include <libassert/assert.hpp>
#include <stdexcept>

void libassert_throw_exception_handler(libassert::assertion_info const &info) {
  throw std::runtime_error("Assertion failed:\n" + info.to_string());
}

int main(int argc, char **argv) {
  libassert::set_failure_handler(libassert_throw_exception_handler);

  return doctest::Context(argc, argv).run();
}
