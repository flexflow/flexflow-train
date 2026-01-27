#include "utils/containers/make_counter_func.h"
#include <memory>

namespace FlexFlow {

struct Counter {
public:
  Counter() = delete;
  Counter(int start) : next_val(std::make_shared<int>(start)) {}

  int operator()() {
    int result = *this->next_val;
    (*this->next_val)++;
    return result;
  }

private:
  std::shared_ptr<int> next_val;
};

std::function<int()> make_counter_func(int start) {
  return Counter{start};
}

} // namespace FlexFlow
