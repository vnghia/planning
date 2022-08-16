#pragma once

enum class Reward { linear_2, convex_2 };

static constexpr auto linear_reward_2 = [](const auto& costs, const auto& state,
                                           auto offset) {
  return -(costs(0, state[offset + 0]) * state[0] +
           costs(1, state[offset + 1]) * state[1]);
};

static constexpr auto convex_reward_2 = [](const auto& costs, const auto& state,
                                           auto offset, auto cost_eps) {
  return -(costs(0, state[offset + 0]) * state[0] +
           costs(1, state[offset + 1]) *
               (state[1] * state[1] * cost_eps + state[1]));
};
