#pragma once

#include <array>
#include <cmath>
#include <limits>
#include <random>
#include <ranges>
#include <typeinfo>
#include <utility>

#include "Fastor/Fastor.h"

using IT = int;

template <IT n_queue, typename F, IT... max_ls>
class Env {
 public:
  using ArrayQF = Fastor::Tensor<F, n_queue>;
  using ArrayQI = Fastor::Tensor<IT, n_queue>;
  using ArrayQB = Fastor::Tensor<bool, n_queue>;
  using STDArrayQI = std::array<IT, n_queue>;
  using TensorQF = Fastor::Tensor<F, max_ls + 1 ..., n_queue>;
  using TensorQI = Fastor::Tensor<IT, max_ls + 1 ..., n_queue>;

 private:
  template <typename T, typename L, IT... i>
  static constexpr auto to_index(const T& a, const L& l,
                                 std::integer_sequence<IT, i...>) {
    return std::make_tuple(a(i)..., l);
  }

  template <IT current, STDArrayQI l, IT... before, IT... after>
  static constexpr auto gen_inf_index(std::integer_sequence<IT, before...>,
                                      std::integer_sequence<IT, after...>) {
    return std::make_tuple(Fastor::fseq<0, l[before] + 1>{}...,
                           Fastor::fseq<0, 1>{},
                           Fastor::seq(0, l[after + current + 1] + 1)...,
                           Fastor::fseq<current, current + 1>{});
  }

  template <STDArrayQI l, IT... i>
  static constexpr auto gen_inf_indices(std::integer_sequence<IT, i...> is) {
    return std::make_tuple(gen_inf_index<i, l>(
        std::make_integer_sequence<IT, i>{},
        std::make_integer_sequence<IT, is.size() - i - 1>{})...);
  }

  template <IT... i>
  static constexpr auto gen_index_0(std::integer_sequence<IT, i...> is) {
    std::array<IT, is.size()> a;
    std::fill(a.begin(), a.end(), 0);
    return std::make_tuple(a[i]...);
  }

  static constexpr STDArrayQI max_ls_a_ = {max_ls...};
  static constexpr auto iota_queue_ = ([]() {
    STDArrayQI a;
    std::iota(a.begin(), a.end(), 0);
    return a;
  })();
  static constexpr auto index_queue_ =
      std::make_integer_sequence<IT, n_queue>{};
  static constexpr auto Q_indices_ = gen_inf_indices<max_ls_a_>(index_queue_);
  static constexpr auto index_queue_1_ =
      std::make_integer_sequence<IT, n_queue + 1>{};

  static constexpr IT n_transition_ = 2 * n_queue + 1;
  using ArrayTF = Fastor::Tensor<F, n_transition_>;
  using ArrayTI = Fastor::Tensor<F, n_transition_>;
  static const inline auto transitions_ = ([]() {
    Fastor::Tensor<IT, n_queue, n_queue> tu;
    tu.zeros();
    Fastor::diag(tu) = 1;

    Fastor::Tensor<IT, n_queue, n_queue> td;
    td.zeros();
    Fastor::diag(td) = -1;

    Fastor::Tensor<IT, n_transition_, n_queue> result;
    result(Fastor::fseq<0, n_queue>{}, Fastor::all) = tu;
    result(Fastor::fseq<n_queue, 2 * n_queue>{}, Fastor::all) = td;
    result(n_transition_ - 1, Fastor::all) = 0;
    return result;
  })();

  static const inline ArrayQI actions_ = iota_queue_;

  ArrayQI states_;

  TensorQF Q_;
  TensorQF n_visit_;

  std::mt19937_64 rng;
  std::uniform_real_distribution<F> eps_dis;

 protected:
  ArrayQF cs_;
  ArrayQF pus_;
  ArrayQF pds_;

  static inline const ArrayQI max_ls_ = max_ls_a_;

 public:
  Env(const ArrayQF& cs, const ArrayQF& pus, const ArrayQF& pds)
      : cs_(cs), pus_(pus), pds_(pds) {
    ResetTrain();
    Reset();
  }

  void ResetTrain() {
    Q_.zeros();
    std::apply(
        [this](auto&&... indices) {
          ((std::apply(this->Q_, indices) =
                -std::numeric_limits<F>::infinity()),
           ...);
        },
        Q_indices_);
    std::apply(Q_, gen_index_0(index_queue_1_)) = 0;

    n_visit_.zeros();
  }

  void Reset(std::mt19937_64::result_type seed = 42) {
    rng.seed(seed);
    states_.zeros();
  }

  virtual F Reward() const {
    return -Fastor::inner(cs_.template cast<F>(), states_.template cast<F>());
  }

  virtual ArrayTF Prob(IT action) {
    ArrayQB allow_u = (states_ < max_ls_);
    auto pus = pus_ * allow_u.template cast<F>();

    ArrayQB allow_d = (action == actions_) && (states_ > 0);
    auto pds = pds_ * allow_d.template cast<F>();

    ArrayTF p;
    p(Fastor::fseq<0, n_queue>{}) = pus;
    p(Fastor::fseq<n_queue, 2 * n_queue>{}) = pds;
    p(2 * n_queue) = 1 - Fastor::sum(pus) - Fastor::sum(pds);
    return p;
  }

  std::pair<ArrayQI, F> Step(IT action) {
    auto p = Prob(action);
    auto idx =
        std::discrete_distribution<IT>(p.data(), p.data() + n_transition_)(rng);
    ArrayQI t = transitions_(idx, Fastor::all);
    return std::make_pair(states_ + t, Reward());
  }

  void Train(F gamma = 0.9, F eps = 0.01, F decay = 0.5, IT epoch = 1,
             IT ls = 1000000, F lr_pow = 0.51) {
    ResetTrain();

    for (IT i = 0; i < epoch; ++i) {
      Reset();

      for (IT j = 0; j < ls; ++j) {
        IT a{};
        ArrayQB m_as = states_ != 0;
        if (Fastor::any_of(m_as)) {
          std::vector<IT> idx_as;
          for (IT k = 0; k < n_queue; ++k) {
            if (m_as(k)) {
              idx_as.push_back(k);
            }
          }

          if (eps_dis(rng) < eps) {
            a = idx_as[std::uniform_int_distribution<IT>(
                0, idx_as.size() - 1)(rng)];
          } else {
            auto max_it = std::max_element(
                iota_queue_.begin(), iota_queue_.end(),
                [this](IT largest, IT current) {
                  return std::apply(this->Q_, to_index(this->states_, largest,
                                                       index_queue_)) <
                         std::apply(this->Q_, to_index(this->states_, current,
                                                       index_queue_));
                });
            a = *max_it;
          }
        }

        auto [nstates, reward] = Step(a);

        auto max_nit = std::max_element(
            iota_queue_.begin(), iota_queue_.end(),
            [this, nstates = &nstates](IT largest, IT current) {
              return std::apply(this->Q_,
                                to_index(*nstates, largest, index_queue_)) <
                     std::apply(this->Q_,
                                to_index(*nstates, current, index_queue_));
            });
        F nq = std::apply(Q_, to_index(nstates, *max_nit, index_queue_));

        const auto st_index = to_index(states_, a, index_queue_);

        const F lr = std::pow(std::apply(n_visit_, st_index) + 1, -lr_pow);
        std::apply(Q_, st_index) +=
            lr * (reward + gamma * nq - std::apply(Q_, st_index));
        std::apply(n_visit_, st_index) += 1;
        states_ = nstates;
      }
    }
  }

  const TensorQF& Q() { return Q_; }
  const TensorQI& n_visit() { return n_visit_; }
};
