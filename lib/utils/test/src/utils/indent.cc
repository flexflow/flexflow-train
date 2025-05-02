#include <doctest/doctest.h>
#include "utils/indent.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("indent") {
    SUBCASE("string is empty") {
      std::string input = "";
      
      std::string result = indent(input);
      std::string correct = "  ";

      CHECK(result == correct);
    }

    SUBCASE("string is one line") {
      std::string input = "hello world";
      std::string result = indent(input);
      std::string correct = "  hello world";

      CHECK(result == correct);
    }

    SUBCASE("string has multiple lines") {
      std::string input = "\n"
                          "a b\n"
                          "c d\n"
                          "e f\n"
                          "g\n";

      std::string result = indent(input);
      std::string correct = "  \n"
                            "  a b\n"
                            "  c d\n"
                            "  e f\n"
                            "  g\n"
                            "  ";

      CHECK(result == correct);
    }

    SUBCASE("leading and trailing whitespace is preserved") {
      std::string input = "   a b  \n"
                          "c   d e\n"
                          "     ";

      std::string result = indent(input);
      std::string correct = "     a b  \n"
                            "  c   d e\n"
                            "       ";

      CHECK(result == correct);
    }
  }
}
