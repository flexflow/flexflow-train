#include "pcg/metric.h"

namespace FlexFlow {
MetricsAttrs::MetricsAttrs(LossFunction _loss_type,
                          std::vector<Metric> const &metrics)
  : loss_type(_loss_type), measure_accuracy(false),
    measure_categorical_crossentropy(false),
    measure_sparse_categorical_crossentropy(false),
    measure_mean_squared_error(false), measure_root_mean_squared_error(false),
    measure_mean_absolute_error(false) {
for (Metric const &m : metrics) {
  switch (m) {
    case Metric::ACCURACY:
      measure_accuracy = true;
      continue;
    case Metric::CATEGORICAL_CROSSENTROPY:
      measure_categorical_crossentropy = true;
      continue;
    case Metric::SPARSE_CATEGORICAL_CROSSENTROPY:
      measure_sparse_categorical_crossentropy = true;
      continue;
    case Metric::MEAN_SQUARED_ERROR:
      measure_mean_squared_error = true;
      continue;
    case Metric::ROOT_MEAN_SQUARED_ERROR:
      measure_root_mean_squared_error = true;
      continue;
    case Metric::MEAN_ABSOLUTE_ERROR:
      measure_mean_absolute_error = true;
      continue;
    default:
      throw mk_runtime_error("Initializing MetricsAttrs with unrecogonized metrics type");
  }
}
}

  
}
