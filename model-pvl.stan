functions {
    real[] pvl(int[] x, int T, real A, real theta) {
        real r[T];
        real mem;
        mem = 0;
        for (t in 1:T) {
            r[t] = theta*mem;
            mem = A*mem + 2*x[t] - 1;
        }
        return r;
    }
}
data {
    int<lower=0> T; // Number of trials
    int<lower=0> N; // Number of participants
    int<lower=0, upper=1> x[N,T];
    int<lower=0, upper=1> y[N,T];
}
parameters {
    vector[2] pvl_params[N]; // A, theta
    cholesky_factor_corr[2] L_Omega;
    vector[2] mu;
    vector<lower=0>[2] scale;
}
transformed parameters {
    real A[N];
    real theta[N];
    matrix[2,2] sigma;
    real nu;

    for (i in 1:N) {
        A[i] = inv_logit(pvl_params[i,1]);
        theta[i] = exp(pvl_params[i,2]);
    }
    sigma = diag_pre_multiply(scale, L_Omega);
    sigma = sigma * sigma';
    nu = 4;
}
model {
    mu ~ normal(0, 100);
    scale ~ normal(0, 1);

    L_Omega ~ lkj_corr_cholesky(1);
    pvl_params ~ multi_student_t(nu, mu, sigma);

    for (i in 1:N) {
        y[i] ~ bernoulli_logit(pvl(x[i], T, A[i], theta[i]));
    }
}
