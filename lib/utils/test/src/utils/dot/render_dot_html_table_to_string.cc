#include "utils/dot/render_dot_html_table_to_string.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("render_dot_html_table_to_string") {
    DotHtmlTable input = DotHtmlTable{
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

    std::string result = render_dot_html_table_to_string(input);
    std::string correct = "<TABLE BORDER=\"0\" CELLBORDER=\"1\" "
                          "CELLSPACING=\"0\"><TR><TD>5</TD></TR></TABLE>";

    CHECK(result == correct);
  }
}
