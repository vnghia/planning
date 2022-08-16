#include "planning/state.hpp"

CartesianProduct::CartesianProduct(const VectorAI& lens)
    : lens(lens), n(lens.prod()), d(lens.cols()), a(([this]() {
        ArrayI a;
        a.resize(n, d);
        for (index_type i = 0; i < n; ++i) {
          auto cur = i;
          for (index_type j = 0; j < d; ++j) {
            const auto rj = d - 1 - j;
            a(i, rj) = cur % this->lens[rj];
            cur /= this->lens[rj];
          }
        }
        return a;
      })()){};

bool CartesianProduct::operator==(const CartesianProduct& other) const {
  return (lens == other.lens).all() && (n == other.n) && (d == other.d) &&
         (a == other.a).all();
}

CartesianProduct::CartesianProduct(VectorAI&& lens, index_type&& n,
                                   index_type&& d, ArrayI&& a)
    : lens(lens), n(n), d(d), a(a) {}

void CartesianProduct::save(cereal::BinaryOutputArchive& ar) const {
  ar(lens, n, d, a);
}

void CartesianProduct::load(cereal::BinaryInputArchive& ar) {
  VectorAI lens;

  index_type n;
  index_type d;

  ArrayI a;

  ar(lens, n, d, a);

  new (this) CartesianProduct(std::move(lens), std::move(n), std::move(d),
                              std::move(a));
}

State::State(index_type n_env, const VectorAI& limits)
    : n_env(n_env),
      n_cls(limits.cols()),
      cls_dims(limits + 1),
      cls(limits + 1),
      env(VectorAI::Constant(n_cls, n_env)),
      sys((VectorAI(cls.d + env.d) << cls.lens, env.lens).finished()),
      lin_spaced(VectorAI::LinSpaced(sys.n, 0, sys.n - 1)),
      to_cls(lin_spaced / env.n),
      to_env(lin_spaced.unaryExpr(
          [this](const index_type x) { return x % env.n; })),
      to_sys(lin_spaced.reshaped<storage_order>(cls.n, env.n)) {}

index_type State::to_cls_action(index_type i, index_type a) const {
  return i * n_cls + a;
}

index_type State::to_sys_action(index_type i, index_type a) const {
  return i * n_cls + a;
}

std::optional<index_type> State::next_to(const ConstRowXprAI& s1,
                                         const ConstRowXprAI& s2,
                                         index_type a) const {
  const auto diff = s2 - s1;
  if (diff.count() != 1) {
    return std::nullopt;
  }
  for (index_type i = 0; i < sys.d; ++i) {
    if (diff(i) &&
        ((i >= n_cls) || ((diff(i) == -1 && a == i) || diff(i) == 1))) {
      return i;
    }
  }
  return std::nullopt;
}

bool State::operator==(const State& other) const {
  return (n_env == other.n_env) && (n_cls == other.n_cls) &&
         (cls_dims == other.cls_dims).all() && (cls == other.cls) &&
         (env == other.env) && (sys == other.sys) &&
         (lin_spaced == other.lin_spaced).all() &&
         (to_cls == other.to_cls).all() && (to_env == other.to_env).all() &&
         (to_sys == other.to_sys).all();
}

State::State(index_type&& n_env, index_type&& n_cls, VectorAI&& cls_dims,
             CartesianProduct&& cls, CartesianProduct&& env,
             CartesianProduct&& sys, VectorAI&& lin_spaced, VectorAI&& to_cls,
             VectorAI&& to_env, ArrayI&& to_sys)
    : n_env(n_env),
      n_cls(n_cls),
      cls_dims(cls_dims),
      cls(cls),
      env(env),
      sys(sys),
      lin_spaced(lin_spaced),
      to_cls(to_cls),
      to_env(to_env),
      to_sys(to_sys) {}

void State::save(cereal::BinaryOutputArchive& ar) const {
  ar(n_env, n_cls, cls_dims, cls, env, sys, lin_spaced, to_cls, to_env, to_sys);
}

void State::load(cereal::BinaryInputArchive& ar) {
  index_type n_env;
  index_type n_cls;

  VectorAI cls_dims;

  CartesianProduct cls;
  CartesianProduct env;
  CartesianProduct sys;

  VectorAI lin_spaced;

  VectorAI to_cls;
  VectorAI to_env;
  ArrayI to_sys;

  ar(n_env, n_cls, cls_dims, cls, env, sys, lin_spaced, to_cls, to_env, to_sys);

  new (this) State(std::move(n_env), std::move(n_cls), std::move(cls_dims),
                   std::move(cls), std::move(env), std::move(sys),
                   std::move(lin_spaced), std::move(to_cls), std::move(to_env),
                   std::move(to_sys));
}
