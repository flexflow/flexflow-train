#include "flexflow/config.h"
#include "flexflow/machine_view.h"
#include "gtest/gtest.h"
#include <iostream>

using namespace Legion;
using namespace FlexFlow;

TEST(machine_view_get_domain, basic) {
  MachineView mv;
  mv.ndims = 1;
  mv.start_device_id = 2;
  mv.dim[0] = 2;
  mv.stride[0] = 1;

  Domain d;
  d.dim = 1;
  d.rect_data[0] = 0;
  d.rect_data[0 + d.dim] =
      1; // Domain is includes, MachineView is exclusive on hi

  EXPECT_EQ(mv.get_domain(), d);
}

TEST(machine_view_get_device_id, basic) {
  MachineView mv;
  mv.ndims = 1;
  mv.start_device_id = 2;
  mv.dim[0] = 2;
  mv.stride[0] = 1;

  EXPECT_EQ(mv.get_device_id({0}), 2);
  EXPECT_EQ(mv.get_device_id({1}), 3);
}

TEST(machine_view_hash, basic) {
  MachineView mv1;
  mv1.ndims = 1;
  mv1.start_device_id = 2;
  mv1.dim[0] = 2;
  mv1.stride[0] = 1;

  MachineView mv2;
  mv2.ndims = 1;
  mv2.start_device_id = 2;
  mv2.dim[0] = 2;
  mv2.stride[0] = 1;

  EXPECT_EQ(mv1.hash(), mv2.hash());
}

TEST(machine_view_hash, different_device_type) {
  MachineView mv1;
  mv1.device_type = MachineView::GPU;
  mv1.ndims = 1;
  mv1.start_device_id = 2;
  mv1.dim[0] = 2;
  mv1.stride[0] = 1;

  MachineView mv2;
  mv2.device_type = MachineView::CPU;
  mv2.ndims = 1;
  mv2.start_device_id = 2;
  mv2.dim[0] = 2;
  mv2.stride[0] = 1;

  EXPECT_NE(mv1.hash(), mv2.hash());
}

TEST(machine_view_hash, different_ndims) {
  MachineView mv1;
  mv1.ndims = 1;
  mv1.start_device_id = 2;
  mv1.dim[0] = 2;
  mv1.stride[0] = 1;

  MachineView mv2;
  mv2.ndims = 2;
  mv2.start_device_id = 2;
  mv2.dim[0] = 2;
  mv2.stride[0] = 1;

  EXPECT_NE(mv1.hash(), mv2.hash());
}

TEST(machine_view_hash, different_start_device_id) {
  MachineView mv1;
  mv1.ndims = 1;
  mv1.start_device_id = 2;
  mv1.dim[0] = 2;
  mv1.stride[0] = 1;

  MachineView mv2;
  mv2.ndims = 1;
  mv2.start_device_id = 3;
  mv2.dim[0] = 2;
  mv2.stride[0] = 1;

  EXPECT_NE(mv1.hash(), mv2.hash());
}

TEST(machine_view_hash, different_dim) {
  MachineView mv1;
  mv1.ndims = 1;
  mv1.start_device_id = 2;
  mv1.dim[0] = 2;
  mv1.stride[0] = 1;

  MachineView mv2;
  mv2.ndims = 1;
  mv2.start_device_id = 2;
  mv2.dim[0] = 3;
  mv2.stride[0] = 1;

  EXPECT_NE(mv1.hash(), mv2.hash());
}

TEST(machine_view_hash, different_stride) {
  MachineView mv1;
  mv1.ndims = 1;
  mv1.start_device_id = 2;
  mv1.dim[0] = 2;
  mv1.stride[0] = 1;

  MachineView mv2;
  mv2.ndims = 1;
  mv2.start_device_id = 2;
  mv2.dim[0] = 2;
  mv2.stride[0] = 2;

  EXPECT_NE(mv1.hash(), mv2.hash());
}

TEST(machine_view_hash, known_collision) {
  MachineView mv1;
  mv1.device_type = MachineView::GPU;
  mv1.ndims = 1;
  mv1.start_device_id = 0;
  mv1.dim[0] = 32;
  mv1.stride[0] = 1;
  
  MachineView mv2;
  mv2.device_type = MachineView::GPU;
  mv2.ndims = 1;
  mv2.start_device_id = 1;
  mv2.dim[0] = 1;
  mv2.stride[0] = 1;
  std::size_t h2 = mv2.hash();

  EXPECT_NE(mv1.hash(), mv2.hash());
}
