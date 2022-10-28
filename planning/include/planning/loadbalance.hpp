#pragma once

#include <iostream>
#include <string>
#include <vector>

#include "Eigen/Dense"
#include "Eigen/SparseCore"
#include "planning/config.hpp"
#include "planning/state.hpp"
#include "planning/system.hpp"
#include "unsupported/Eigen/CXX11/Tensor"

class LoadBalance : public System {
 public:
  LoadBalance(const index_type n_env, const index_type n_cls,
              const VectorAI& limits, const float_type* costs,
              const float_type arrival, const float_type* departures,
              const float_type* env_trans_mats,
              const reward_func_type& reward_func,
              const std::optional<float_type>& normalized_c = std::nullopt);
  LoadBalance(const System& system) : System(system) {}
};
