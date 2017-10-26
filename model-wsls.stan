functions {
    real[] wsls(int[] x, int[] y, int T, real pw0, real pl0, real thetaw, real thetal) {
        real p1[T]; // Probability of responding 1
        real pw;
        real pl;

        pw = pw0;
        pl = pl0;
        p1[1] = 0.5; // No win/loss has occurred in the first trial

        for (t in 2:T) {
            if (x[t - 1] == y[t - 1]) { // The previous trial was a win
                if (y[t - 1] == 1)
                    p1[t] = pw;
                else
                    p1[t] = (1 - pw);
                pw = pw + thetaw*(1 - pw);
                pl = (1 - thetal)*pl;
            }
            else {
                if (y[t - 1] == 1)
                    p1[t] = (1 - pl);
                else
                    p1[t] = pl;
                pl = pl + thetal*(1 - pl);
                pw = (1 - thetaw)*pw;
            }
        }
        return p1;
    }
}
data {
    int<lower=0> T; // Number of trials
    int<lower=0> N; // Number of participants
    int<lower=0, upper=1> x[N,T];
    int<lower=0, upper=1> y[N,T];
    // Cross validation
    int<lower=0> N_; // Number of participants for CV
    int<lower=0, upper=1> x_[N_,T];
    int<lower=0, upper=1> y_[N_,T];
}
parameters {
    vector[4] params[N];
    cholesky_factor_corr[4] L_Omega;
    vector[4] mu;
    vector<lower=0>[4] scale;
}
transformed parameters {
    matrix[4,4] sigma;
    real nu;

    sigma = diag_pre_multiply(scale, L_Omega);
    sigma = sigma * sigma';
    nu = 4;
}
model {
    mu ~ normal(0, 5);
    scale ~ cauchy(0, 1);

    L_Omega ~ lkj_corr_cholesky(1);
    params ~ multi_student_t(nu, mu, sigma);

    for (i in 1:N) {
        y[i] ~ bernoulli(wsls(x[i], y[i], T, inv_logit(params[i,1]), inv_logit(params[i,2]), inv_logit(params[i,3]), inv_logit(params[i,4])));
    }
}
generated quantities {
    real log_lik[N_];
    for (i in 1:N_) {
        vector[4] params_;
        params_ = multi_student_t_rng(nu, mu, sigma);
        log_lik[i] = bernoulli_lpmf(y_[i] | wsls(x_[i], y_[i], T, inv_logit(params_[1]), inv_logit(params_[2]), inv_logit(params_[3]), inv_logit(params_[4])));
    }
}
