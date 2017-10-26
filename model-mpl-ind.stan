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
    int<lower=0, upper=1> x[T];
    int<lower=0, upper=1> y[T];
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
    real<lower=0, upper=1> A;
    real<lower=0, upper=1> rho;
    real<lower=0, upper=5> theta;
}
transformed parameters {
    real Axrho;
    vector[kmaxp1] ps;
    
    Axrho = A*rho;
    for (kp1 in 1:kmaxp1) {
        ps[kp1] = log(1./kmaxp1) + bernoulli_logit_lpmf(y |
            mpl(x, T, kp1 - 1, pow2[kp1], A, rho, theta));
    }
}
model {
    target += log_sum_exp(ps);    
}
generated quantities {
    int k;
    vector[kmaxp1] probk;
    {
        real q;
        q = log_sum_exp(ps);
        for (kp1 in 1:kmaxp1) {
            probk[kp1] = exp(ps[kp1] - q);
        }
    }
    k = categorical_logit_rng(ps) - 1;
}