#include "utils/graph/cow_ptr_t.h"

namespace FlexFlow {

struct MyClonableStruct {
  MyClonableStruct *clone() const; 
};

template struct cow_ptr_t<MyClonableStruct>;

} // namespace FlexFlow
