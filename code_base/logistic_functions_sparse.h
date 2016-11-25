int logistic_sparse_flag = 1;
// Without otherwise specify:

/// *Xt - data matrix
/// *y - set of labels
/// n - number of sample
/// d - dimension of the data
/// lambda - regularization parameter
/// *w - test point

double compute_info_sparse(double *Xt, double *w, double* y, long idx,
	mwIndex *ir, mwIndex *jc)
{
	double tmp = 0;
	// Inner product
	for (long j = jc[idx]; j < jc[idx+1]; j++) {
		tmp += w[ir[j]] * Xt[j];
	}
	tmp = exp(-y[idx] * tmp);
	tmp = y[idx] / (1+tmp);
	return tmp;
}

void compute_partial_gradient_sparse(double *Xt, double *w, double *y, double *g,
	long n, long d, double lambda, long i, mwIndex *ir, mwIndex *jc)
{
	// Initialize the gradient
	for (long j = 0; j < d; j++) {
		g[j] = 0;
	}

	// Sum the gradients of individual functions
	double sigmoid = compute_info_sparse(Xt, w, y, i, ir, jc);
	for (long j = jc[i]; j < jc[i+1]; j++) {
		g[ir[j]] += (sigmoid - y[i]) * Xt[j];
	}
}

void compute_full_gradient_sparse(double *Xt, double *w, double *y, double *g,
	long n, long d, double lambda, mwIndex *ir, mwIndex *jc)
{
	// Initialize the gradient
	for (long i = 0; i < d; i++) {
		g[i] = 0;
	}

	// Sum the gradients of individual functions
	double sigmoid;
	for (long i = 0; i < n; i++) {
		sigmoid = compute_info_sparse(Xt, w, y, i, ir, jc);
		for (long j = jc[i]; j < jc[i+1]; j++) {
			g[ir[j]] += (sigmoid - y[i]) * Xt[j];
		}
	}

	// Average the gradients and add gradient of regularizer
	for (long i = 0; i < d; i++) {
		g[i] = g[i] / n;
	}
}


/// Compute the function value of average regularized logistic loss
double compute_function_value_sparse(double* w, double *Xt, double *y,
	long n, long d, double lambda, mwIndex *ir, mwIndex *jc)
{
	double value = 0;
	double tmp;
	// Compute losses of individual functions and average them
	for (long i = 0; i < n; i++) {
		tmp = 0;
		for (long j = jc[i]; j < jc[i+1]; j++) {
			tmp += Xt[j] * w[ir[j]];
		}
		value += log(1 + exp(-y[i] * tmp));
	}
	value = value / n;

	// Add regularization term
	for (long j = 0; j < d; j++) {
		value += (lambda / 2) * w[j] * w[j];
	}
	return value;
}
