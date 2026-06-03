#include "utils/dot/dot_html_from_json.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("dot_html_table_from_json") {
    SUBCASE("json is int") {
      nlohmann::json j = 5;

      DotHtmlTable result = dot_html_table_from_json(j);
      DotHtmlTable correct = DotHtmlTable{
          /*border=*/0_n,
          /*cellborder=*/1_n,
          /*cellspacing=*/0_n,
          /*rows=*/
          {
              DotHtmlTableRow{
                  /*cells=*/{
                      DotHtmlTableCell{
                          /*contents=*/DotHtmlTableCellContents{
                              std::string{"5"},
                          },
                          /*port=*/std::nullopt,
                          /*colspan=*/std::nullopt,
                      },
                  },
              },
          },
      };

      CHECK(result == correct);
    }

    SUBCASE("json is array") {
      nlohmann::json j = std::vector<int>{
          3,
          5,
          4,
          3,
          2,
      };

      auto mk_row = [](std::string const &x) -> DotHtmlTableRow {
        return DotHtmlTableRow{
            /*cells=*/{
                DotHtmlTableCell{
                    /*contents=*/DotHtmlTableCellContents{
                        x,
                    },
                    /*port=*/std::nullopt,
                    /*colspan=*/std::nullopt,
                },
            },
        };
      };

      DotHtmlTable result = dot_html_table_from_json(j);
      DotHtmlTable correct = DotHtmlTable{
          /*border=*/0_n,
          /*cellborder=*/1_n,
          /*cellspacing=*/0_n,
          /*rows=*/
          {
              mk_row("3"),
              mk_row("5"),
              mk_row("4"),
              mk_row("3"),
              mk_row("2"),
          },
      };

      CHECK(result == correct);
    }

    SUBCASE("json is object") {
      nlohmann::json j;

      j["hello"] = 3;
      j["world"] = "yes";

      auto mk_kv_row = [](std::string const &k,
                          std::string const &v) -> DotHtmlTableRow {
        return DotHtmlTableRow{
            /*cells=*/{
                DotHtmlTableCell{
                    /*contents=*/DotHtmlTableCellContents{
                        k,
                    },
                    /*port=*/std::nullopt,
                    /*colspan=*/std::nullopt,
                },
                DotHtmlTableCell{
                    /*contents=*/DotHtmlTableCellContents{
                        v,
                    },
                    /*port=*/std::nullopt,
                    /*colspan=*/std::nullopt,
                },
            },
        };
      };

      DotHtmlTable result = dot_html_table_from_json(j);
      DotHtmlTable correct = DotHtmlTable{
          /*border=*/0_n,
          /*cellborder=*/1_n,
          /*cellspacing=*/0_n,
          /*rows=*/
          {
              mk_kv_row("hello", "3"),
              mk_kv_row("world", "yes"),
          },
      };

      CHECK(result == correct);
    }

    SUBCASE("json is nested objects") {
      nlohmann::json j;

      j["hello"] = 3;
      j["world"] = "yes";
      j["two"] = nlohmann::json{
          {"abc", 5},
          {"def", "no"},
      };
      j["red"] = nlohmann::json{
          {"blue", "green"},
      };

      auto mk_kv_row =
          [](std::string const &k,
             DotHtmlTableCellContents const &v) -> DotHtmlTableRow {
        return DotHtmlTableRow{
            /*cells=*/{
                DotHtmlTableCell{
                    /*contents=*/DotHtmlTableCellContents{
                        k,
                    },
                    /*port=*/std::nullopt,
                    /*colspan=*/std::nullopt,
                },
                DotHtmlTableCell{
                    /*contents=*/v,
                    /*port=*/std::nullopt,
                    /*colspan=*/std::nullopt,
                },
            },
        };
      };

      DotHtmlTable result = dot_html_table_from_json(j);
      DotHtmlTable correct = DotHtmlTable{
          /*border=*/0_n,
          /*cellborder=*/1_n,
          /*cellspacing=*/0_n,
          /*rows=*/
          {
              mk_kv_row("hello", DotHtmlTableCellContents{"3"}),
              mk_kv_row(
                  "red",
                  DotHtmlTableCellContents{
                      DotHtmlTable{
                          /*border=*/0_n,
                          /*cellborder=*/1_n,
                          /*cellspacing=*/0_n,
                          /*rows=*/
                          {
                              mk_kv_row("blue",
                                        DotHtmlTableCellContents{"green"}),
                          },
                      },
                  }),
              mk_kv_row(
                  "two",
                  DotHtmlTableCellContents{
                      DotHtmlTable{
                          /*border=*/0_n,
                          /*cellborder=*/1_n,
                          /*cellspacing=*/0_n,
                          /*rows=*/
                          {
                              mk_kv_row("abc", DotHtmlTableCellContents{"5"}),
                              mk_kv_row("def", DotHtmlTableCellContents{"no"}),
                          },
                      },
                  }),
              mk_kv_row("world", DotHtmlTableCellContents{"yes"}),
          },
      };

      CHECK(result == correct);
    }
  }
}
