#ifndef _FF_METRICS_H_
#define _FF_METRICS_H_

#include <unordered_set>
#include "utils/fmt.h"
#include "op-attrs/ops/loss_functions/loss_functions.h"

namespace FlexFlow {

enum class Metric {
  ACCURACY,
  CATEGORICAL_CROSSENTROPY,
  SPARSE_CATEGORICAL_CROSSENTROPY,
  MEAN_SQUARED_ERROR,
  ROOT_MEAN_SQUARED_ERROR,
  MEAN_ABSOLUTE_ERROR,
};

class MetricsAttrs {
public:
  MetricsAttrs() = delete;
  MetricsAttrs(LossFunction, std::vector<Metric> const &);

public:
  LossFunction loss_type;
  bool measure_accuracy;
  bool measure_categorical_crossentropy;
  bool measure_sparse_categorical_crossentropy;
  bool measure_mean_squared_error;
  bool measure_root_mean_squared_error;
  bool measure_mean_absolute_error;
};

} // namespace FlexFlow

namespace fmt {

template <>
struct formatter<::FlexFlow::Metric> : formatter<string_view> {
  template <typename FormatContext>
  auto format(::FlexFlow::Metric m, FormatContext &ctx) const
      -> decltype(ctx.out()) {
    using namespace FlexFlow;

    string_view name = "unknown";
    switch (m) {
      case Metric::ACCURACY:
        name = "Accuracy";
        break;
      case Metric::CATEGORICAL_CROSSENTROPY:
        name = "CategoricalCrossEntropy";
        break;
      case Metric::SPARSE_CATEGORICAL_CROSSENTROPY:
        name = "SparseCategoricalCrossEntropy";
        break;
      case Metric::MEAN_SQUARED_ERROR:
        name = "MeanSquaredError";
        break;
      case Metric::ROOT_MEAN_SQUARED_ERROR:
        name = "RootMeanSquaredError";
        break;
      case Metric::MEAN_ABSOLUTE_ERROR:
        name = "MeanAbsoluteError";
        break;
    }
    return formatter<string_view>::format(name, ctx);
  }
};

} // namespace fmt


#endif
