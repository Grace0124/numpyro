#ifndef STAN_MATH_PRIM_PROB_CONCRETE_LPDF_HPP
#define STAN_MATH_PRIM_PROB_CONCRETE_LPDF_HPP

#include <cmath>
#include <stan/math.hpp>
#include <stan/math/prim/err.hpp>
#include <stan/math/prim/fun/as_column_vector_or_scalar.hpp>
#include <stan/math/prim/fun/constants.hpp>
#include <stan/math/prim/fun/dot_product.hpp>
#include <stan/math/prim/fun/eval.hpp>
#include <stan/math/prim/fun/lgamma.hpp>
#include <stan/math/prim/fun/log.hpp>
#include <stan/math/prim/fun/log_determinant_ldlt.hpp>
#include <stan/math/prim/fun/max_size_mvt.hpp>
#include <stan/math/prim/fun/mdivide_left_ldlt.hpp>
#include <stan/math/prim/fun/size_mvt.hpp>
#include <stan/math/prim/fun/sum.hpp>
#include <stan/math/prim/fun/to_ref.hpp>
#include <stan/math/prim/fun/transpose.hpp>
#include <stan/math/prim/fun/vector_seq_view.hpp>
#include <stan/math/prim/functor/partials_propagator.hpp>
#include <stan/math/prim/meta.hpp>

namespace concrete_model_namespace {

template <bool propto, typename T_y, typename T_alpha, typename T_lambda,
          stan::require_all_not_nonscalar_prim_or_rev_kernel_expression_t<
              T_y, T_alpha, T_lambda>* = nullptr>
stan::return_type_t<T_y, T_alpha, T_lambda> concrete_lpdf(
    const T_y& y, const T_alpha& alpha, const T_lambda& lambda,
    std::ostream* pstream__) {
  using stan::is_constant;
  using stan::is_constant_all;
  using stan::partials_return_t;
  using stan::ref_type_t;
  using stan::scalar_seq_view;
  using stan::vector_seq_view;
  using stan::math::max_size;
  using stan::math::size;
  using namespace stan::math;

  using T_partials_return = partials_return_t<T_y, T_alpha, T_lambda>;
  using T_partials_array = typename Eigen::Array<T_partials_return, -1, -1>;
  using T_y_ref = ref_type_t<T_y>;
  using T_alpha_ref = ref_type_t<T_alpha>;
  using T_lambda_ref = ref_type_t<T_lambda>;

  static constexpr const char* function = "concrete_lpdf";
  check_consistent_sizes_mvt(function, "y", y, "mu", alpha);
  if (size_mvt(y) == 0 || size_mvt(alpha) == 0 || size_zero(lambda)) {
    return 0.0;
  }

  T_y_ref y_ref = y;
  T_alpha_ref alpha_ref = alpha;
  T_lambda_ref lambda_ref = lambda;
  vector_seq_view<T_y_ref> y_vec(y_ref);
  vector_seq_view<T_alpha_ref> alpha_vec(alpha_ref);
  scalar_seq_view<T_lambda_ref> lambda_vec(lambda_ref);
  const int t_length = max_size_mvt(y, alpha);
  const int t_size = y_vec[0].size();

  check_consistent_sizes(function, "Random variable", y_vec[0],
                         "Location parameter", alpha_vec[0]);
  for (size_t i = 1; i < t_length; i++) {
    check_size_match(
        function, "Size of one of the vectors of the location variable",
        alpha_vec[i].size(),
        "Size of the first vector of the location variable", t_size);
    check_size_match(function,
                     "Size of one of the vectors of the random variable",
                     y_vec[i].size(),
                     "Size of the first vector of the random variable", t_size);
  }

  for (size_t t = 0; t < t_length; t++) {
    check_positive(function, "Location parameter", alpha_vec[t]);
    check_simplex(function, "Random variable", y_vec[t]);
  }
  check_positive(function, "Temperature parameter", lambda_ref);

  if (!include_summand<propto, T_y, T_alpha, T_lambda>::value) {
    return 0.0;
  }

  T_partials_array y_dbl(t_size, t_length);
  for (size_t t = 0; t < t_length; t++) {
    y_dbl.col(t) = y_vec.val(t);
  }
  T_partials_array alpha_dbl(t_size, t_length);
  for (size_t t = 0; t < t_length; t++) {
    alpha_dbl.col(t) = alpha_vec.val(t);
  }
  T_partials_array lambda_dbl(1, t_length);
  for (size_t t = 0; t < t_length; t++) {
    lambda_dbl.col(t) = lambda_vec.val(t);
  }

  auto ops_partials = make_partials_propagator(y_ref, alpha_ref, lambda_ref);
  T_partials_return logp(0);

  T_partials_array S =
      (alpha_dbl * (y_dbl.pow(-lambda_dbl.replicate(t_size, 1))))
          .colwise()
          .sum();

  // printf("t_size: %d\n", t_size);
  // printf("t_length: %d\n", t_length);
  // std::cout << "y_dbl:\n" << y_dbl << std::endl;
  // std::cout << "alpha_dbl:\n" << alpha_dbl << std::endl;
  // std::cout << "lambda_dbl:\n" << lambda_dbl << std::endl;
  // std::cout << "S:\n" << S << std::endl;

  if (include_summand<propto>::value) {
    logp += t_length * stan::math::lgamma(t_size);
  }
  if (include_summand<propto, T_lambda>::value) {
    logp += (t_size - 1) * lambda_dbl.log().sum();
  }
  if (include_summand<propto, T_alpha>::value) {
    logp += alpha_dbl.log().sum();
  }
  if (include_summand<propto, T_y, T_lambda>::value) {
    logp += -((lambda_dbl + 1).replicate(t_size, 1) * y_dbl.log()).sum();
  }
  if (include_summand<propto, T_y, T_alpha, T_lambda>::value) {
    logp += -t_size * S.log().sum();
  }

  if (!is_constant_all<T_y>::value) {
    for (size_t t = 0; t < t_length; t++) {
      T_partials_return S_j = S(0, t);
      T_partials_return lambda_j = lambda_dbl(0, t);
      T_partials_array p1 =
          -(lambda_j + 1) / y_dbl.col(t) +
          t_size * lambda_j / S_j *
              (y_dbl.col(t).pow(-lambda_j - 1) * alpha_dbl.col(t));
      // std::cout << "p1: " << p1 << std::endl;
      partials_vec<0>(ops_partials)[t] += p1.matrix();
    }
  }
  if (!is_constant_all<T_alpha>::value) {
    for (size_t t = 0; t < t_length; t++) {
      T_partials_return S_j = S(0, t);
      T_partials_return lambda_j = lambda_dbl(0, t);
      T_partials_array p2 =
          1 / alpha_dbl.col(t) - t_size * y_dbl.col(t).pow(-lambda_j) / S_j;
      // std::cout << "p2: " << p2 << std::endl;
      partials_vec<1>(ops_partials)[t] += p2.matrix();
    }
  }
  if (!is_constant_all<T_lambda>::value) {
    for (size_t t = 0; t < t_length; t++) {
      T_partials_return S_j = S(0, t);
      T_partials_return lambda_j = lambda_dbl(0, t);
      T_partials_return p3 =
          (t_size - 1) / lambda_j - y_dbl.col(t).log().sum() +
          t_size / S_j *
              (y_dbl.col(t).log() * y_dbl.col(t).pow(-lambda_j) *
               alpha_dbl.col(t))
                  .sum();
      // std::cout << "p3: " << p3 << std::endl;
      partials<2>(ops_partials)[t] += p3;
    }
  }

  return ops_partials.build(logp);
}

template <typename T_y, typename T_alpha, typename T_lambda>
stan::return_type_t<T_y, T_alpha, T_lambda> concrete_lpdf(
    const T_y& y, const T_alpha& alpha, const T_lambda& lambda,
    std::ostream* pstream__) {
  return concrete_lpdf<false>(y, alpha, lambda);
}

}  // namespace concrete_model_namespace
#endif
