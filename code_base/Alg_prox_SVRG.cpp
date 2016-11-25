#include "config.h"

using namespace std;
/*
	USAGE:
	hist = SVRG(w, Xt, y, lambda, Lmax, iters_outer);
	==================================================================
	INPUT PARAMETERS:
	w (d x 1) - initial point; updated in place
	Xt (d x n) - data matrix; transposed (data points are columns); real
	y (n x 1) - labels; in {-1,1}
	lambda - scalar regularization param
	Lmax - 
	iters_outer
	==================================================================
	OUTPUT PARAMETERS:
	hist = array of function values after each outer loop.
		   Computed ONLY if explicitely asked for output in MATALB.
*/

/// SVRG runs the SVRG algorithm for solving regularized 
/// logistic regression on dense data provided
/// nlhs - number of output parameters requested
///		   if set to 1, function values are computed
/// *prhs[] - array of pointers to the input arguments
mxArray* SVRG_dense(int nlhs, const mxArray *prhs[]) {

	//////////////////////////////////////////////////////////////////
	/// Declare variables ////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////

	// Input variables
	double *w, *Xt, *y;
	double lambda, Lmax, iters_outer, alpha;	

	// Other variables
	long i, j, k; // Some loop indexes
	long n, d; // Dimensions of problem
	long iters_inner; // Number of inner loops

	bool evalf = false; // set to true if function values should be evaluated

	double *mu;
	double *g_prev;
	double *g_tilda;
	double *w_prev;
	double *w_tilda;

	double *hist; // Used to store function value at points in history

	mxArray *plhs; // History array to return if needed

	//////////////////////////////////////////////////////////////////
	/// Process input ////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////

	w = mxGetPr(prhs[0]); // The variable to be learned
	Xt = mxGetPr(prhs[1]); // Data matrix (transposed)
	y = mxGetPr(prhs[2]); // Labels
	lambda = mxGetScalar(prhs[3]); // Regularization parameter
	Lmax = mxGetScalar(prhs[4]); // Lmax (constant)
	iters_outer = mxGetScalar(prhs[5]); // outer loops (constant)


	if (nlhs == 1) {
		evalf = true;
	}
	
	//////////////////////////////////////////////////////////////////
	/// Get problem related constants ////////////////////////////////
	//////////////////////////////////////////////////////////////////

	d = mxGetM(prhs[1]); // Number of features, or dimension of problem
	n = mxGetN(prhs[1]); // Number of samples, or data points
	iters_inner = 2 * n; // Number of inner iterations
	alpha = 1 / (Lmax);

	std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, n-1);

	//////////////////////////////////////////////////////////////////
	/// Initialize some values ///////////////////////////////////////
	//////////////////////////////////////////////////////////////////

	mu = new double[d];
	g_prev = new double[d];
	g_tilda = new double[d];
	w_prev = new double[d];
	w_tilda = new double[d];
	if (evalf == true) {
		plhs = mxCreateDoubleMatrix(iters_outer + 1, 1, mxREAL);
		hist = mxGetPr(plhs);
	}

	// Initiate w
	for (j = 0; j < d; j++) {
		w_prev[j] = 0;
		w_tilda[j] = w_prev[j];
	}

	// The outer loop
	for (k = 0; k < iters_outer; k++)
	{
		// Evaluate function value if output requested
		if (evalf == true) {		
			hist[k] = compute_function_value(w_tilda, Xt, y, n, d, lambda);
		}

		// mu with the full gradient 
		compute_full_gradient(Xt, w_tilda, y, mu, n, d, lambda);

		// The inner loop
		for (i = 0; i < iters_inner; i++) {
			long idx = dis(gen);
			// Compute gradient of last inner iter
			compute_partial_gradient(Xt, w_prev, y, g_prev, n, d, lambda, idx);
			
			// Compute the gradient of last outer iter
			compute_partial_gradient(Xt, w_tilda, y, g_tilda, n, d, lambda, idx);

			// Update the test point 
			update_test_point_SVRG(w_prev, g_prev, g_tilda, mu, alpha, d);

			prox_map(w_prev, d, lambda);
		}

		for (j = 0; j < d; j++) {
			w_tilda[j] = w_prev[j];
		}

	}

    for (j = 0; j < d; j++) {
		w[j] = w_tilda[j];
	}
    
	//////////////////////////////////////////////////////////////////
	/// Free some memory /////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////

	delete[] mu;
	delete[] g_prev;
	delete[] g_tilda;
	delete[] w_prev;
	delete[] w_tilda;

	if (evalf == true) { return plhs; }
	else { return 0; }
}

/// SVRG runs the SVRG algorithm for solving regularized 
/// logistic regression on sparse data provided
/// nlhs - number of output parameters requested
///		   if set to 1, function values are computed
/// *prhs[] - array of pointers to the input arguments
mxArray* SVRG_sparse(int nlhs, const mxArray *prhs[]) {

	//////////////////////////////////////////////////////////////////
	/// Declare variables ////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////

	// Input variables
	double *w, *Xt, *y;
	double lambda, Lmax, iters_outer, alpha;	

	// Other variables
	long i, j, k; // Some loop indexes
	long n, d; // Dimensions of problem
	long iters_inner; // Number of inner loops
	double info, info_prev;

	bool evalf = false; // set to true if function values should be evaluated

	long *last_seen;
	double *mu;
	double *g_prev;
	double *g_tilda;
	double *w_prev;
	double *w_tilda;

	double *hist; // Used to store function value at points in history

	mwIndex *ir, *jc; // Used to access nonzero elements of Xt
	mxArray *plhs; // History array to return if needed

	//////////////////////////////////////////////////////////////////
	/// Process input ////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////

	w = mxGetPr(prhs[0]); // The variable to be learned
	Xt = mxGetPr(prhs[1]); // Data matrix (transposed)
	y = mxGetPr(prhs[2]); // Labels
	lambda = mxGetScalar(prhs[3]); // Regularization parameter
	Lmax = mxGetScalar(prhs[4]); // Lmax (constant)
	iters_outer = mxGetScalar(prhs[5]); // outer loops (constant)


	if (nlhs == 1) {
		evalf = true;
	}
	
	//////////////////////////////////////////////////////////////////
	/// Get problem related constants ////////////////////////////////
	//////////////////////////////////////////////////////////////////

	d = mxGetM(prhs[1]); // Number of features, or dimension of problem
	n = mxGetN(prhs[1]); // Number of samples, or data points
	jc = mxGetJc(prhs[1]); // pointers to starts of columns of Xt
	ir = mxGetIr(prhs[1]); // row indexes of individual elements of Xt
	iters_inner = 2 * n; // Number of inner iterations
	alpha = 1 / (Lmax*n/10);
	
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<> dis(0, n-1);

	//////////////////////////////////////////////////////////////////
	/// Initialize some values ///////////////////////////////////////
	//////////////////////////////////////////////////////////////////

	mu = new double[d];
	g_prev = new double[d];
	g_tilda = new double[d];
	w_prev = new double[d];
	w_tilda = new double[d];
	last_seen = new long[d];
	if (evalf == true) {
		plhs = mxCreateDoubleMatrix(iters_outer + 1, 1, mxREAL);
		hist = mxGetPr(plhs);
	}


	//////////////////////////////////////////////////////////////////
	/// The SVRG algorithm ///////////////////////////////////////////
	//////////////////////////////////////////////////////////////////

	// The outer loop
	for (k = 0; k < iters_outer; k++)
	{
		// Evaluate function value if output requested
		if (evalf == true) {		
			hist[k] = compute_function_value_sparse(w, Xt, y, n, d, lambda, ir, jc);
		}

		// mu with the full gradient 
		compute_full_gradient_sparse(Xt, w, y, mu, n, d, lambda, ir, jc);
		// Initiation
		for (j = 0; j < d; j++) {
			w_prev[j] = w[j];
			last_seen[j] = 0;
			mu[j] += lambda * w[j];
		}

		// The inner loop
		for (i = 0; i < iters_inner; i++) {
			long idx = dis(gen);

			for (long j = jc[idx]; j < jc[idx+1]; j++) {
				w[ir[j]] -= alpha * (i - last_seen[ir[j]]) * mu[ir[j]];
				last_seen[ir[j]] = i;

				// Proximal Mapping
				if (w[ir[j]] >= lambda) { w[ir[j]] -= lambda; }
				else if (w[ir[j]] <= -lambda ) { w[ir[j]] += lambda; }
				else { w[ir[j]] = 0; }
			}

			info = compute_info_sparse(Xt, w, y, idx, ir, jc);
			info_prev = compute_info_sparse(Xt, w_prev, y, idx, ir, jc);

			// Update the test point 
			update_test_point_sparse_SVRG(Xt + jc[idx], w, y[idx], info,
				info_prev, jc[idx + 1] - jc[idx], alpha, ir + jc[idx]);
		}
		for (long j = 0; j < d; j++) {
			w[j] -= alpha * (iters_inner - last_seen[j]) * mu[j];
		}
	}
	
	//////////////////////////////////////////////////////////////////
	/// Free some memory /////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////

	delete[] mu;
	delete[] g_prev;
	delete[] g_tilda;
	delete[] w_prev;
	delete[] w_tilda;
	delete[] last_seen;

	if (evalf == true) { return plhs; }
	else { return 0; }
}

/// Entry function of MATLAB
/// nlhs - number of output parameters
/// *plhs[] - array poiters to the outputs
/// nrhs - number of input parameters
/// *prhs[] - array of pointers to inputs
/// For more info about this syntax see 
/// http://www.mathworks.co.uk/help/matlab/matlab_external/gateway-routine.html
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

	// First determine, whether the data matrix is stored in sparse format.
	// If it is, use more efficient algorithm
	if (mxIsSparse(prhs[1])) {
		cout << "sparse" << endl;
		plhs[0] = SVRG_sparse(nlhs, prhs);
	}
	else {
		plhs[0] = SVRG_dense(nlhs, prhs);
	}
}
