#include "kernels/perf_metrics.h"
#include "doctest/doctest.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("Test PerfMetrics Constructors and Metric Functions") {
    SUBCASE("Test constructor with start_time only") {
      double start = 100.0;
      PerfMetrics pm(start);

      CHECK(pm.start_time == start);
      CHECK(pm.current_time == start);

      CHECK(pm.train_all == 0);
      if (pm.train_correct.has_value()) {
        CHECK(pm.train_correct.value() == 0);
      }
      
      CHECK(!pm.cce_loss.has_value());
      
      if (pm.sparse_cce_loss.has_value()) {
        CHECK(pm.sparse_cce_loss.value() == doctest::Approx(0.0f));
      }
      if (pm.mse_loss.has_value()) {
        CHECK(pm.mse_loss.value() == doctest::Approx(0.0f));
      }
      if (pm.rmse_loss.has_value()) {
        CHECK(pm.rmse_loss.value() == doctest::Approx(0.0f));
      }
      if (pm.mae_loss.has_value()) {
        CHECK(pm.mae_loss.value() == doctest::Approx(0.0f));
      }
    }

    SUBCASE("Test full constructor and throughput/accuracy") {
      int train_all = 200;
      int train_correct = 150;
      float cce = 1.2f;
      float sparse_cce = 1.0f;
      float mse = 0.5f;
      float rmse = 0.7f;
      float mae = 0.3f;
      double start = 100.0;
      double curr = 110.0;
      PerfMetrics pm(train_all,
                     train_correct,
                     cce,
                     sparse_cce,
                     mse,
                     rmse,
                     mae,
                     start,
                     curr);

      CHECK(pm.train_all == train_all);
      CHECK(pm.train_correct.has_value());
      CHECK(pm.train_correct.value() == train_correct);
      CHECK(pm.cce_loss.has_value());
      CHECK(pm.cce_loss.value() == doctest::Approx(cce));
      CHECK(pm.sparse_cce_loss.has_value());
      CHECK(pm.sparse_cce_loss.value() == doctest::Approx(sparse_cce));
      CHECK(pm.mse_loss.has_value());
      CHECK(pm.mse_loss.value() == doctest::Approx(mse));
      CHECK(pm.rmse_loss.has_value());
      CHECK(pm.rmse_loss.value() == doctest::Approx(rmse));
      CHECK(pm.mae_loss.has_value());
      CHECK(pm.mae_loss.value() == doctest::Approx(mae));
      CHECK(pm.start_time == start);
      CHECK(pm.current_time == curr);

      float expected_throughput = train_all / (curr - start);
      CHECK(get_throughput(pm) == doctest::Approx(expected_throughput));

      float expected_accuracy = static_cast<float>(train_correct) / train_all;
      CHECK(get_accuracy(pm) == doctest::Approx(expected_accuracy));
    }

    SUBCASE("Test update function") {
      PerfMetrics pm1(100, 50, 1.0f, 0.5f, 0.3f, 0.2f, 0.1f, 0.0, 1.0);
      PerfMetrics pm2(50, 30, 0.5f, 0.3f, 0.2f, 0.1f, 0.05f, 0.0, 1.5);

      PerfMetrics updated = update(pm1, pm2);

      CHECK(updated.train_all == (100 + 50));
      if (updated.train_correct.has_value()) {
        CHECK(updated.train_correct.value() == (50 + 30));
      }

      CHECK(updated.cce_loss.has_value());
      CHECK(updated.cce_loss.value() == doctest::Approx(1.0f + 0.5f));
      CHECK(updated.sparse_cce_loss.has_value());
      CHECK(updated.sparse_cce_loss.value() == doctest::Approx(0.5f + 0.3f));
      CHECK(updated.mse_loss.has_value());
      CHECK(updated.mse_loss.value() == doctest::Approx(0.3f + 0.2f));
      CHECK(updated.rmse_loss.has_value());
      CHECK(updated.rmse_loss.value() == doctest::Approx(0.2f + 0.1f));
      CHECK(updated.mae_loss.has_value());
      CHECK(updated.mae_loss.value() == doctest::Approx(0.1f + 0.05f));
      CHECK(updated.current_time == pm2.current_time);
    }

    SUBCASE("Test apply_scale function") {
      PerfMetrics pm(100, 50, 2.0f, 1.0f, 0.8f, 0.6f, 0.4f, 0.0, 2.0);
      float scale = 0.5f;
      PerfMetrics scaled = apply_scale(pm, scale);

      CHECK(scaled.cce_loss.has_value());
      CHECK(scaled.cce_loss.value() == doctest::Approx(2.0f * scale));
      CHECK(scaled.sparse_cce_loss.has_value());
      CHECK(scaled.sparse_cce_loss.value() == doctest::Approx(1.0f * scale));
      CHECK(scaled.mse_loss.has_value());
      CHECK(scaled.mse_loss.value() == doctest::Approx(0.8f * scale));
      CHECK(scaled.rmse_loss.has_value());
      CHECK(scaled.rmse_loss.value() == doctest::Approx(0.6f * scale));
      CHECK(scaled.mae_loss.has_value());
      CHECK(scaled.mae_loss.value() == doctest::Approx(0.4f * scale));

      CHECK(scaled.train_all == pm.train_all);
      if (scaled.train_correct.has_value()) {
        CHECK(scaled.train_correct.value() == pm.train_correct.value());
      }
      CHECK(scaled.start_time == pm.start_time);
      CHECK(scaled.current_time == pm.current_time);
    }
  }
}
