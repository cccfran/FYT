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

/// Update the test point *w in place once you have everything prepared
/// *g_tilda - gradient of w_tilda
void update_test_point_SVRG(double *w_prev, double *g_prev, 
	double *g_tilda, double *mu, double alpha, long d)
{
	for (long i = 0; i < d; i++) {
		w_prev[i] = w_prev[i] - alpha * (g_prev[i] - g_tilda[i] + mu[i]);
	}
}

/// Update the test point *w in place once you have everything prepared
/// *g_table - gradient table
void update_test_point_SAGA(double *w_prev, double *g_prev, 
	double *g_new, double *g_table, double alpha, long n, long d)
{
	double tmp_mean_gradient;
	for (long j = 0; j < d; j++) {
		for (long i = 0; i < n; i++) {
			tmp_mean_gradient += *(g_table + i*d + j);
		}
		tmp_mean_gradient /= n;
		w_prev[j] = w_prev[j] - alpha * (g_new[j] - g_prev[j] + tmp_mean_gradient);
	}
}


void prox_map(double *w, long d, double t){
// Soft Thresholding
//  tmp = (abs(w)-t);
//  tmp = (tmp+abs(tmp))/2;
//  y   = sign(w).*tmp;
	double *tmp;
	tmp = new double[d];
	
	for(int i = 0; i < d; i ++){
		tmp[i] = abs(w[i]) - t;
	}

	for(int i = 0; i < d; i ++){
		tmp[i] = (tmp[i] + abs(tmp[i]))/2;
	}

	for(int i = 0; i < d; i ++){
		int sign = (0 < w[i]) - (w[i] < 0);
		w[i] = sign * tmp[i];
	}

	delete [] tmp;
}
