// Without otherwise specify:

/// n - number of sample
/// d - dimension of the data
/// alpha - learning rate
/// mu - average of aggregate gradient 
/// *w_prev - test point; updated in place
/// *g_prev - gradient of last step


#include <iostream>
#include <cmath>
using namespace std;

// Updates the test point *w in place
/// Makes the step only in the nonzero coordinates of *x,
/// and without regularizer. The regularizer step is constant
/// across more iterations --- updated in lazy_updates
/// *x - training example
/// *w - test point; updated in place
/// sigmoid - sigmoid at current point *w
/// sigmoidold - sigmoid at old point *wold
/// d - number of nonzeros of training example *x
/// alpha - learning rate
/// *ir - row indexes of nonzero elements of *x
void update_test_point_sparse_SVRG(double *x, double *w, double y,
	double sigmoid, double sigmoidold,
	long d, double alpha, mwIndex *ir)
{
	for (long j = 0; j < d; j++) {
		w[ir[j]] -= alpha * (x[j] * (sigmoid - sigmoidold));
	}
}
