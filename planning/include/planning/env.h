#pragma once

#include <array>
#include <cmath>
#include <limits>
#include <random>
#include <utility>
#include <vector>

#include "Fastor/Fastor.h"

using int_type = int;

template <int_type n_queue_t, typename F, bool save_qs_t, int_type... max_ls>
class Env {
 public:
  static constexpr int_type n_queue = n_queue_t;
  static constexpr int_type n_transition = 2 * n_queue + 1;
  static constexpr int_type n_total = ((max_ls + 1) * ...) * n_queue;
  static constexpr bool save_qs = save_qs_t;

  using float_type = F;

  using array_fq_type = Fastor::Tensor<float_type, n_queue>;
  using array_iq_type = Fastor::Tensor<int_type, n_queue>;
  using array_bq_type = Fastor::Tensor<bool, n_queue>;

  using std_vector_f_type = std::vector<float_type>;
  using std_array_i_type = std::array<int_type, n_queue>;

  using tensor_f_type = Fastor::Tensor<float_type, max_ls + 1 ..., n_queue>;
  using tensor_i_type = Fastor::Tensor<int_type, max_ls + 1 ..., n_queue>;
  using tensor_r_type = Fastor::Tensor<float_type, max_ls + 1 ...>;

  using array_ft_type = Fastor::Tensor<float_type, n_transition>;
  using array_it_type = Fastor::Tensor<float_type, n_transition>;

  static constexpr std_array_i_type dim_ls = {max_ls + 1 ...};

  static constexpr auto iota_nq = ([]() {
    std_array_i_type a;
    std::iota(a.begin(), a.end(), 0);
    return a;
  })();

  template <int_type current, std_array_i_type l, int_type... before,
            int_type... after>
  static constexpr auto gen_inf_index(
      std::integer_sequence<int_type, before...>,
      std::integer_sequence<int_type, after...>) {
    return std::make_tuple(Fastor::fseq<0, l[before]>{}...,
                           Fastor::fseq<0, 1>{},
                           Fastor::fseq<0, l[after + current + 1]>{}...,
                           Fastor::fseq<current, current + 1>{});
  }

  template <std_array_i_type l, int_type... i>
  static constexpr auto gen_inf_indices(
      std::integer_sequence<int_type, i...> is) {
    return std::make_tuple(gen_inf_index<i, l>(
        std::make_integer_sequence<int_type, i>{},
        std::make_integer_sequence<int_type, is.size() - i - 1>{})...);
  }

  static constexpr auto idx_nq =
      std::make_integer_sequence<int_type, n_queue>{};
  static constexpr auto idx_nq_1 =
      std::make_integer_sequence<int_type, n_queue + 1>{};
  static constexpr auto inf_idxs = gen_inf_indices<dim_ls>(idx_nq);

 public:
  Env(const std_vector_f_type& cs, const std_vector_f_type& pus,
      const std_vector_f_type& pds)
      : cs_(cs), pus_(pus), pds_(pds) {}

  void init_once() { this->reward_mat_ = this->init_reward(); }

  void reset_train() {
    q_.zeros();
    std::apply(
        [this](auto&&... indices) {
          ((std::apply(this->q_, indices) =
                -std::numeric_limits<float_type>::infinity()),
           ...);
        },
        inf_idxs);
    std::apply(q_, gen_index_0(idx_nq_1)) = 0;

    n_visit_.zeros();
  }

  void reset(std::mt19937_64::result_type seed = 42) {
    rng.seed(seed);
    states_.zeros();
  }

  virtual array_ft_type prob(int_type action) {
    array_bq_type allow_u = (states_ < max_ls_);
    array_fq_type pus = pus_ * allow_u.template cast<float_type>();

    array_bq_type allow_d = (action == actions_) && (states_ > 0);
    array_fq_type pds = pds_ * allow_d.template cast<float_type>();

    array_ft_type p;
    p(Fastor::fseq<0, n_queue>{}) = pus;
    p(Fastor::fseq<n_queue, 2 * n_queue>{}) = pds;
    p(Fastor::fix<2 * n_queue>) = 1 - Fastor::sum(pus) - Fastor::sum(pds);
    return p;
  }

  array_iq_type step(int_type action) {
    auto p = prob(action);
    auto idx = std::discrete_distribution<int_type>(
        p.data(), p.data() + n_transition)(rng);
    array_iq_type t = transitions_(idx, Fastor::all);
    return states_ + t;
  }

  void train(float_type gamma = 0.9, float_type eps = 0.01,
             float_type decay = 0.5, int_type epoch = 1, uint64_t ls = 20000000,
             float_type lr_pow = 0.51, std::mt19937_64::result_type seed = 42) {
    reset_train();

    if constexpr (save_qs) {
      qs_.resize(epoch * ls * n_total);
    }

    for (int_type i = 0; i < epoch; ++i) {
      reset(seed);

      for (int_type j = 0; j < ls; ++j) {
        int_type a{};
        array_bq_type as = states_ != 0;
        if (Fastor::any_of(as)) {
          std::vector<int_type> idx_as;
          for (int_type k = 0; k < n_queue; ++k) {
            if (as(k)) {
              idx_as.push_back(k);
            }
          }

          if (eps_dis(rng) < eps) {
            a = idx_as[std::uniform_int_distribution<int_type>(
                0, idx_as.size() - 1)(rng)];
          } else [[likely]] {
            auto max_it = std::max_element(
                iota_nq.begin(), iota_nq.end(),
                [this](int_type largest, int_type current) {
                  return std::apply(this->q_,
                                    to_index(this->states_, largest, idx_nq)) <
                         std::apply(this->q_,
                                    to_index(this->states_, current, idx_nq));
                });
            a = *max_it;
          }
        }

        auto nstates = step(a);

        auto max_nit = std::max_element(
            iota_nq.begin(), iota_nq.end(),
            [this, &nstates](int_type largest, int_type current) {
              return std::apply(this->q_, to_index(nstates, largest, idx_nq)) <
                     std::apply(this->q_, to_index(nstates, current, idx_nq));
            });
        float_type nq = std::apply(q_, to_index(nstates, *max_nit, idx_nq));

        const auto st_index = to_index(states_, a, idx_nq);

        auto reward = std::apply(reward_mat_, to_index(nstates, idx_nq));
        const float_type lr =
            std::pow(std::apply(n_visit_, st_index) + 1, -lr_pow);
        std::apply(q_, st_index) +=
            lr * (reward + gamma * nq - std::apply(q_, st_index));

        if constexpr (save_qs) {
          std::copy(q_.data(), q_.data() + n_total,
                    qs_.begin() + i * ls + j * n_total);
        }

        std::apply(n_visit_, st_index) += 1;
        states_ = nstates;
      }
    }
  }

  void from_array(const float_type* q, const int_type* n_visit, float_type* qs,
                  size_t qs_size) {
    q_ = q;
    n_visit_ = n_visit;
    if constexpr (save_qs) {
      std::copy(qs, qs + qs_size, qs_.begin());
    }
  }

  const tensor_f_type& q() const { return q_; }
  const tensor_i_type& n_visit() const { return n_visit_; }
  const std_vector_f_type& qs() const { return qs_; }
  const tensor_r_type& reward_mat() const { return reward_mat_; }

 protected:
  array_fq_type cs_;
  array_fq_type pus_;
  array_fq_type pds_;
  array_iq_type states_;

  static constexpr array_iq_type max_ls_ = {max_ls...};

 private:
  virtual tensor_r_type init_reward() const { return {}; }

  static const inline array_iq_type actions_ = iota_nq;

  tensor_f_type q_;
  tensor_i_type n_visit_;
  std_vector_f_type qs_;
  tensor_r_type reward_mat_;

  std::mt19937_64 rng;
  std::uniform_real_distribution<float_type> eps_dis;

  template <typename T, typename L, int_type... i>
  static constexpr auto to_index(const T& a, const L& l,
                                 std::integer_sequence<int_type, i...>) {
    return std::make_tuple(a(i)..., l);
  }

  template <typename T, int_type... i>
  static constexpr auto to_index(const T& a,
                                 std::integer_sequence<int_type, i...>) {
    return std::make_tuple(a(i)...);
  }

  template <int_type... i>
  static constexpr auto gen_index_0(std::integer_sequence<int_type, i...> is) {
    return std::make_tuple(Fastor::fix<0 * i>...);
  }

  static const inline auto transitions_ = ([]() {
    Fastor::Tensor<int_type, n_queue, n_queue> tu;
    tu.zeros();
    Fastor::diag(tu) = 1;

    Fastor::Tensor<int_type, n_queue, n_queue> td;
    td.zeros();
    Fastor::diag(td) = -1;

    Fastor::Tensor<int_type, n_transition, n_queue> result;
    result(Fastor::fseq<0, n_queue>{}, Fastor::all) = tu;
    result(Fastor::fseq<n_queue, 2 * n_queue>{}, Fastor::all) = td;
    result(Fastor::fix<n_transition - 1>, Fastor::all) = 0;
    return result;
  })();
};

template <typename F, bool save_qs_t, int_type... max_ls>
class LinearEnv : public Env<2, F, save_qs_t, max_ls...> {
 public:
  using parent_type = Env<2, F, save_qs_t, max_ls...>;
  using parent_type::dim_ls;
  using parent_type::parent_type;

  typename parent_type::tensor_r_type init_reward() const override {
    typename parent_type::tensor_r_type res;
    for (int_type i = 0; i < dim_ls[0]; ++i) {
      for (int_type j = 0; j < dim_ls[1]; ++j) {
        res(i, j) = -(cs_[0] * i + cs_[1] * j);
      }
    }
    return res;
  }

 protected:
  using parent_type::cs_;
};

#include <iostream>

template <typename F, bool save_qs_t, int_type... max_ls>
class ConvexEnv : public Env<2, F, save_qs_t, max_ls...> {
 public:
  using parent_type = Env<2, F, save_qs_t, max_ls...>;
  using parent_type::dim_ls;
  using parent_type::parent_type;

  typename parent_type::tensor_r_type init_reward() const override {
    typename parent_type::tensor_r_type res;
    for (int_type i = 0; i < dim_ls[0]; ++i) {
      for (int_type j = 0; j < dim_ls[1]; ++j) {
        res(i, j) = -(cs_[0] * i + cs_[1] * j * j);
      }
    }
    return res;
  }

 protected:
  using parent_type::cs_;
};
