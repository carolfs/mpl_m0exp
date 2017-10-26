functions {
    real[] mpl(int[] x, int T, int k, int num_mem, real A, real rho, real theta) {
        real r[T];
        if (k == 0) {
            real mem;
            mem = 0;
            for (t in 1:T) {
                r[t] = theta*mem;
                mem = rho*A*mem + 2*x[t] - 1;
            }
        }
        else {
            vector[num_mem] mem;
            int eta;
            for (i in 1:num_mem) {
                mem[i] = 0;
            }
            eta = 1;
            for (t in 1:k) {
                eta = 2*(eta - 1) + x[t] + 1;
                r[t] = 0;
            }
            for (t in (k+1):T) {
                r[t] = theta*mem[eta];
                for (i in 1:num_mem) {
                    mem[i] = A*mem[i];
                }
                mem[eta] = rho*mem[eta] + 2*x[t] - 1;
                eta = (2*(eta - 1) + x[t]) % num_mem + 1;
            }
        }
        return r;
    }
}
data {
    int<lower=0> kmaxp1; // maximum k + 1
    int<lower=0> T; // Number of trials
    int<lower=0> N; // Number of participants
    int<lower=0, upper=1> x[N,T];
    int<lower=0, upper=1> y[N,T];
}
transformed data {
    int pow2[kmaxp1];
    vector[kmaxp1] alpha;
    pow2[1] = 1;
    for (i in 2:kmaxp1) {
        pow2[i] = 2*pow2[i-1];
    }
    for (i in 1:kmaxp1) {
        alpha[i] = 0.001;
    }
}
parameters {
    vector[3] mpl_params[N]; // A, rho, theta
    cholesky_factor_corr[3] L_Omega;
    simplex[kmaxp1] probk;
    vector[3] mu;
    vector<lower=0>[3] scale;
}
transformed parameters {
    real A[N];
    real rho[N];
    real theta[N];
    matrix[3,3] sigma;
    real nu;

    for (i in 1:N) {
        A[i] = inv_logit(mpl_params[i,1]);
        rho[i] = inv_logit(mpl_params[i,2]);
        theta[i] = exp(mpl_params[i,3]);
    }
    sigma = diag_pre_multiply(scale, L_Omega);
    sigma = sigma * sigma';
    nu = 4;
}
model {
    real ps[kmaxp1];

    mu ~ normal(0, 100);
    scale ~ normal(0, 1);
    probk ~ dirichlet(alpha);

    L_Omega ~ lkj_corr_cholesky(1);
    mpl_params ~ multi_student_t(nu, mu, sigma);

    for (i in 1:N) {
        for (kp1 in 1:kmaxp1) {
            ps[kp1] = log(probk[kp1]) + bernoulli_logit_lpmf(y[i] |
                mpl(x[i], T, kp1 - 1, pow2[kp1], A[i], rho[i], theta[i]));
        }
        target += log_sum_exp(ps);
    }
}
