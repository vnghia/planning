#include "planning/queuing.hpp"

Queuing::Queuing(const index_type n_env, const index_type n_cls,
                 const VectorAI& limits, const float_type* costs,
                 const float_type* arrivals, const float_type* departures,
                 const float_type* env_trans_mats,
                 const reward_func_type& reward_func,
                 const std::optional<float_type>& normalized_c)
    : System(
          n_env, n_cls, limits,
          Eigen::TensorMap<const Tensor2F>(costs, n_cls, n_env),
          Eigen::TensorMap<const Tensor2F>(arrivals, n_cls, n_env),
          Eigen::TensorMap<const Tensor2F>(departures, n_cls, n_env),
          env_trans_mats, reward_func, normalized_c,
          [](const System& system) {
            return to_scalar(system.arrivals.maximum(std::array{1}).sum() +
                             system.departures.maximum() +
                             system.env_trans_mats.sum(std::array{2})
                                 .maximum(std::array{1})
                                 .sum());
          },
          [&limits](const State& states, index_type i, index_type a) {
            return (states.to_cls(i) && !states.sys.a(i, a)) ||
                   (!states.to_cls(i) && a);
          },
          [n_cls](const State& states, const ConstRowXprAI& s1,
                  const ConstRowXprAI& s2,
                  index_type a) -> std::optional<index_type> {
            const auto diff = s2 - s1;
            if (diff.count() != 1) {
              return std::nullopt;
            }
            for (index_type i = 0; i < states.sys.d; ++i) {
              if (diff(i) && ((i >= n_cls) ||
                              ((diff(i) == -1 && a == i) || diff(i) == 1))) {
                return i;
              }
            }
            return std::nullopt;
          }) {}
