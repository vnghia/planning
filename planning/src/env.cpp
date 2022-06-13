#include "planning/env.h"

#include "nanobind/nanobind.h"
#include "nanobind/stl/vector.h"
#include "nanobind/tensor.h"

namespace nb = nanobind;

using namespace nb::literals;

NB_MODULE(planning_ext, m) {
  using SmallEnv = Env<2, double, 3, 3>;
  auto cls =
      nb::class_<SmallEnv>(m, "SmallEnv")
          .def(
              nb::init<const SmallEnv::STDVectorF&, const SmallEnv::STDVectorF&,
                       const SmallEnv::STDVectorF&>())
          .def("Train", &SmallEnv::Train, "gamma"_a = 0.9, "eps"_a = 0.01,
               "decay"_a = 0.5, "epoch"_a = 1, "ls"_a = 1000000,
               "lr_pow"_a = 0.51);
}
