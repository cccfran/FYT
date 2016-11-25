#include "config.h"
using namespace std;
/*
	USAGE:
	hist = SAGA(w, Xt, y, lambda, Lmax, iVals, iters);
	==================================================================
	INPUT PARAMETERS:
	w (d x 1) - initial point; updated in place
	Xt (d x n) - data matrix; transposed (data points are columns); real
	y (n x 1) - labels; in {-1,1}
	lambda - scalar regularization param
	Lmax - 
	iVals (iters x 1) - random pick sample 
	iters
	==================================================================
	OUTPUT PARAMETERS:
	hist = array of function values after each outer loop.
		   Computed ONLY if explicitely asked for output in MATALB.
*/

/// SAGA runs the SAGA algorithm for solving regularized 
/// logistic regression on dense data provided
/// nlhs - number of output parameters requested
///		   if set to 1, function values are computed
/// *prhs[] - array of pointers to the input arguments
mxArray* SAGA_dense(int nlhs, const mxArray *prhs[]) {

	//////////////////////////////////////////////////////////////////
	/// Declare variables ////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////

	// Input variables
	double *w, *Xt, *y;
	double *minus_y; // hotfix
	double lambda, Lmax, alpha;
	long long* iVals;
	long iters;

	// Other variables
	long i, j, k; // Some loop indexes
	long n, d; // Dimensions of problem
	long idx;
	double grad_info; // logistic: sigmoid; least square: xW - y
	double *infoOld; // infoOld is old grad_info
	double *g; // Direction of update

	bool evalf = false; // set to true if function values should be evaluated

	double *hist; // Used to store function value at points in history

	mwIndex *ir, *jc; // used to access nonzero elements of Xt
	mxArray *plhs; // History array to return if needed

	//////////////////////////////////////////////////////////////////
	/// Process input ////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////

	w = mxGetPr(prhs[0]); // The variable to be learned
	Xt = mxGetPr(prhs[1]); // Data matrix (transposed)
	y = mxGetPr(prhs[2]); // Labels
	lambda = mxGetScalar(prhs[3]); // Regularization parameter
	Lmax = mxGetScalar(prhs[4]); // Lmax (constant)
	iVals = (long long*)mxGetPr(prhs[5]); // Sampled indexes (sampled in advance)
	iters = mxGetScalar(prhs[6]); // outer loops (constant)
	iters = mxGetM(prhs[5]); // Number of outer iterations


	if (nlhs == 1) {
		evalf = true;
	}

	if (!mxIsClass(prhs[5], "int64"))
		mexErrMsgTxt("iVals must be int64");

	//////////////////////////////////////////////////////////////////
	/// Get problem related constants ////////////////////////////////
	//////////////////////////////////////////////////////////////////

	d = mxGetM(prhs[1]); // Number of features, or dimension of problem
	n = mxGetN(prhs[1]); // Number of samples, or data points
	alpha = 1 / (Lmax/10);

	//////////////////////////////////////////////////////////////////
	/// Initialize some values ///////////////////////////////////////
	//////////////////////////////////////////////////////////////////

	// For logistic regression, infoOld is sigmoidOld
	// For least square regression, infoOld is xW - y;
	infoOld = new double[n];
	g = new double[d];
	for (i = 0; i < n; i++) { infoOld[i] = 0; }
	for (i = 0; i < d; i++) { g[i] = 0; }
	if (evalf == true) {
		plhs = mxCreateDoubleMatrix((long)floor((double)iters / (2 * n)) + 1, 1, mxREAL);
		hist = mxGetPr(plhs);
	}
	// Hotfix
	minus_y = new double[n];
	for (i = 0; i < n; i ++) {minus_y[i] = -y[i];}
	// End Hotfix
	
	//////////////////////////////////////////////////////////////////
	/// The SAGA algorithm ///////////////////////////////////////////
	//////////////////////////////////////////////////////////////////

	// The outer loop
	for (k = 0; k < iters; k++)
	{
		if (evalf == true && k % (2 * n) == 0) {
			// Hotfix
			if(logistic_sparse_flag)
				hist[(long)floor((double)k / (2 * n))] = 
					compute_function_value(w, Xt, minus_y, n, d, lambda);
			else
				hist[(long)floor((double)k / (2 * n))] = 
					compute_function_value(w, Xt, y, n, d, lambda);
			// End Hotfix
		}

		idx = *(iVals++); // Sample function and move pointer

		grad_info = compute_info_dense(Xt, w, y, d, idx);

		for (j = 0; j < d; j++) {
			w[j] -= alpha * (Xt[idx*d + j] * (grad_info - infoOld[idx]) + g[j] + lambda * w[j]);
		}
		
		for (j = 0; j < d; j++) {
			g[j] += Xt[idx*d + j] * (grad_info - infoOld[idx]) / n;
		}

		infoOld[idx] = grad_info;
	}

	delete[] infoOld;
	delete[] g;
	// Hotfix
	delete[] minus_y;
	// End Hotfix

	if (evalf == true) { 
		// Hotfix
		for(i = 0; i < d; i ++) {
			w[i] = -w[i];
		}
		// End Hotfix
		return plhs; 		
	}
	else { return 0; }
}

/// SAGA runs the SAGA algorithm for solving regularized 
/// logistic regression on sparse data provided
/// nlhs - number of output parameters requested
///		   if set to 1, function values are computed
/// *prhs[] - array of pointers to the input arguments
mxArray* SAGA_sparse(int nlhs, const mxArray *prhs[]) {

	//////////////////////////////////////////////////////////////////
	/// Declare variables ////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////

	// Input variables
	double *w, *Xt, *y;
	double *minus_y; // hotfix
	double lambda, Lmax, alpha;
	long long* iVals;
	long iters;

	// Other variables
	long i, j, k; // Some loop indexes
	long n, d; // Dimensions of problem
	long idx;
	double grad_info; // logistic: sigmoid; least square: xW - y
	double *infoOld; // infoOld is old grad_info
	double *g; // Direction of update

	bool evalf = false; // set to true if function values should be evaluated

	double *hist; // Used to store function value at points in history
	long *last_seen; // used to do lazy "when needed" updates

	mwIndex *ir, *jc; // used to access nonzero elements of Xt
	mxArray *plhs; // History array to return if needed

	//////////////////////////////////////////////////////////////////
	/// Process input ////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////

	w = mxGetPr(prhs[0]); // The variable to be learned
	Xt = mxGetPr(prhs[1]); // Data matrix (transposed)
	y = mxGetPr(prhs[2]); // Labels
	lambda = mxGetScalar(prhs[3]); // Regularization parameter
	Lmax = mxGetScalar(prhs[4]); // Lmax (constant)
	iVals = (long long*)mxGetPr(prhs[5]); // Sampled indexes (sampled in advance)
	iters = mxGetScalar(prhs[6]); // outer loops (constant)
	iters = mxGetM(prhs[5]); // Number of outer iterations


	if (nlhs == 1) {
		evalf = true;
	}

	if (!mxIsClass(prhs[5], "int64"))
		mexErrMsgTxt("iVals must be int64");

	//////////////////////////////////////////////////////////////////
	/// Get problem related constants ////////////////////////////////
	//////////////////////////////////////////////////////////////////

	d = mxGetM(prhs[1]); // Number of features, or dimension of problem
	n = mxGetN(prhs[1]); // Number of samples, or data points
	jc = mxGetJc(prhs[1]); // pointers to starts of columns of Xt
	ir = mxGetIr(prhs[1]); // row indexes of individual elements of Xt
	alpha = 1 / (Lmax/10);

	//////////////////////////////////////////////////////////////////
	/// Initialize some values ///////////////////////////////////////
	//////////////////////////////////////////////////////////////////

	// For logistic regression, infoOld is sigmoidOld
	// For least square regression, infoOld is xW - y;
	infoOld = new double[n];
	g = new double[d];
	last_seen = new long[d];
	for (i = 0; i < n; i++) { infoOld[i] = 0; }
	for (i = 0; i < d; i++) { g[i] = 0; }
	for (i = 0; i < d; i++) { last_seen[i] = 0; }
	if (evalf == true) {
		plhs = mxCreateDoubleMatrix((long)floor((double)iters / (2 * n)) + 1, 1, mxREAL);
		hist = mxGetPr(plhs);
	}
	// Hotfix
	minus_y = new double[n];
	for (i = 0; i < n; i ++) {minus_y[i] = -y[i];}
	// End Hotfix
	
	//////////////////////////////////////////////////////////////////
	/// The SAGA algorithm ///////////////////////////////////////////
	//////////////////////////////////////////////////////////////////

	// The outer loop
	for (k = 0; k < iters; k++)
	{
		if (evalf == true && k % (2 * n) == 0) {
			// Hotfix
			if(logistic_sparse_flag)
				hist[(long)floor((double)k / (2 * n))] = 
					compute_function_value_sparse(w, Xt, minus_y, n, d, lambda, ir, jc);
			else
				hist[(long)floor((double)k / (2 * n))] = 
					compute_function_value_sparse(w, Xt, y, n, d, lambda, ir, jc);
			// End Hotfix
		}

		idx = *(iVals++); // Sample function and move pointer

		grad_info = compute_info_sparse(Xt, w, y, idx, ir, jc);

		for (j = jc[idx]; j < jc[idx + 1]; j++) {
			w[ir[j]] -= alpha * (Xt[j] * (grad_info - infoOld[idx]) + g[ir[j]] + lambda * w[ir[j]]);
			last_seen[ir[j]] = k;
		}
		
		for (j = jc[idx]; j < jc[idx + 1]; j++) {
			g[ir[j]] += Xt[j] * (grad_info - infoOld[idx]) / n;
		}

		infoOld[idx] = grad_info;
	}

	for (i = 0; i < d; i++) {
		w[i] -= alpha * (iters - last_seen[i]) * (g[i] + lambda * w[i]);
	}

	delete[] infoOld;
	delete[] g;
	delete[] last_seen;
	// Hotfix
	delete[] minus_y;
	// End Hotfix

	if (evalf == true) { 
		// Hotfix
		if (logistic_sparse_flag){
			for(i = 0; i < d; i ++) {
				w[i] = -w[i];
			}
		}
		// End Hotfix
		return plhs; 		
	}
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
		plhs[0] = SAGA_sparse(nlhs, prhs);		
	}
	else {
		plhs[0] = SAGA_dense(nlhs, prhs);
	}
}
