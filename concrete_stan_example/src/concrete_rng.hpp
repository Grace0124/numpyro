#ifndef STAN_MATH_PRIM_PROB_CONCRETE_RNG_HPP
#define STAN_MATH_PRIM_PROB_CONCRETE_RNG_HPP

#include <boost/random/uniform_real_distribution.hpp>
#include <boost/random/variate_generator.hpp>
#include <stan/math/prim/err.hpp>
#include <stan/math/prim/fun/exp.hpp>
#include <stan/math/prim/fun/log.hpp>
#include <stan/math/prim/meta.hpp>
#include <cmath>

namespace concrete_model_namespace {

template <class RNG>
inline Eigen::VectorXd concrete_rng(
    const Eigen::Matrix<double, Eigen::Dynamic, 1>& alpha, const double lambda,
    RNG& rng, std::ostream* pstream__) {
  using boost::variate_generator;
  using boost::random::uniform_real_distribution;
  using Eigen::VectorXd;
  using std::exp;
  using std::log;
  using stan::math::check_positive;

  static constexpr const char* function = "concrete_rng";
  check_positive(function, "Location parameter", alpha);
  check_positive(function, "Temperature parameter", lambda);
  const int n_size = alpha.size();

  variate_generator<RNG&, uniform_real_distribution<> > uniform_rng(
      rng, uniform_real_distribution<>(0.0, 1.0));
  VectorXd weights(n_size);
  for (int i = 0; i < n_size; ++i) {
    double gumble_i = -log(-log(uniform_rng()));
    double logit_i = (log(alpha(i)) + gumble_i) / lambda;
    weights(i) = exp(logit_i);
  }
  return weights / weights.sum();
}

}
#endif
