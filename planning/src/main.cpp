#include <iostream>
#include <vector>

#include "planning/reward.hpp"
#include "planning/system.hpp"

int main() {
  const float_type costs[] = {3, 1, 2, 0};
  const float_type arrivals[] = {0.14, 0.14, 0.14, 0};
  const float_type departures[] = {0.3, 0.34, 0.33, 0};
  const float_type envs[] = {0, 0.01, 0.01, 0, 1, 0, 0, 0};
  auto s =
      System(2, {{5, 5}}, costs, arrivals, departures, envs, linear_reward_2);
  s.train_q_full(0.9, 1, 1000, 42);
}
