functions {
  #include concrete_functions.stan
}
data {
  int N;
  int n;
  array[N] vector[n] y;
}
transformed data {

}
parameters {
  vector<lower=0>[n] alpha;
  real<lower=0> lambda;
}
model {
  alpha ~ lognormal(0, 1);
  lambda ~ lognormal(0, 1);
  target += concrete_lpdf(y | alpha, lambda);
}
generated quantities {
  vector[5] alpha_r = [1, 1, 1, 1, 2]';
  real lambda_r = 1;
  array[1000] vector[5] y_rep;
  for (i in 1:N) {
    y_rep[i] = concrete_rng(alpha_r, lambda_r);
  }

}