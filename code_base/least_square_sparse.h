int logistic_sparse_flag = 0;

double compute_info_sparse(double *Xt, double *w, double* y, long idx, 
    mwIndex *ir, mwIndex *jc)
{
    // Compute xW - y
    double x_i_w = 0;
    for (long j = jc[idx]; j < jc[idx+1]; j++) {
        x_i_w += w[ir[j]] * Xt[j];
    }
    x_i_w = x_i_w - y[idx];
    return x_i_w;
}

void compute_full_gradient_sparse(double *Xt, double *w, double *y, double *g,
    long n, long d, double lambda, mwIndex *ir, mwIndex *jc)
{
    // Init the gradient
    for (long i = 0; i < d; i++) {
        g[i] = 0;
    }

    for (long i = 0; i < n; i++) {
        // Xt(i) dot w;
        double x_i_w = 0;
        for (long j = jc[i]; j < jc[i+1]; j ++){
            x_i_w += Xt[j] * w[ir[j]];
        }
        // Xt(i) dot w - y
        double x_i_w_minus_y = (x_i_w - y[i]) * 2;

        for (long j = jc[i]; j < jc[i+1]; j ++){
            g[ir[j]] += x_i_w_minus_y * Xt[j];
        }
    }

    for (long i = 0; i < d; i++) {
        g[i] = g[i] / n;
    }
}

void compute_partial_gradient_sparse(double *Xt, double *w, double *y, double *g,
    long n, long d, double lambda, long i, mwIndex *ir, mwIndex *jc)
{
    // Init the gradient
    for (long k = 0; k < d; k++) {
        g[k] = 0;
    }

    // Xt(i) dot w;
    double x_i_w = 0;
    for (long j = jc[i]; j < jc[i+1]; j ++){
        x_i_w += Xt[j] * w[ir[j]];
    }
    // Xt(i) dot w - y
    double x_i_w_minus_y = (x_i_w - y[i]);

    for (long j = jc[i]; j < jc[i+1]; j ++){
        g[ir[j]] += x_i_w_minus_y * Xt[j];
    }

    for (long k = 0; k < d; k++) {
        g[k] = g[k] / n;
    }
}


double compute_function_value_sparse(double* w, double *Xt, double *y,
    long n, long d, double lambda, mwIndex *ir, mwIndex *jc)
{
    double value = 0;

    for (long i = 0; i < n; i++) {
        // Xt(i) dot w;
        double x_i_w = 0;
        for (long j = jc[i]; j < jc[i+1]; j ++){
            x_i_w += Xt[j] * w[ir[j]];
        }
        // Xt(i) dot w - y
        double x_i_w_minus_y = x_i_w - y[i];

        // (Xt(i) dot w - y) ^ 2
        value += x_i_w_minus_y * x_i_w_minus_y;
    }
    
    // new
	value = value/(2*n);  

    return value;
}
