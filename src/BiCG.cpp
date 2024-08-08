#include <BiCG.h>

#include <cmath>

#include <omp.h>

inline void mul_CRS_matrix_by_vector(CRSMatrix &A, double *x, double *c) {
#pragma omp parallel for schedule(static)
    for (int i = 0; i < A.n; i++) {
        c[i] = 0.0;
        for (int j = A.rowPtr[i]; j < A.rowPtr[i + 1]; ++j) {
            c[i] += A.val[j] * x[A.colIndex[j]];
        }
    }
}

inline void transpose_CRS_matrix(const CRSMatrix &A, CRSMatrix &A_transpose) {
	A_transpose.n = A.m;
	A_transpose.m = A.n;
	A_transpose.nz = A.nz;

	A_transpose.val.resize(A.nz);
	A_transpose.colIndex.resize(A.nz);
	A_transpose.rowPtr.resize(A.m + 2, 0.0);

	for (int i = 0; i < A.nz; ++i) {
		++A_transpose.rowPtr[A.colIndex[i] + 2];
	}
	for (int i = 1; i <= A.m; ++i) {
		A_transpose.rowPtr[i] += A_transpose.rowPtr[i - 1];
	}

	for (int i = 0; i < A.n; ++i) {
		for (int j = A.rowPtr[i]; j < A.rowPtr[i + 1]; ++j) {
			int index = A_transpose.rowPtr[A.colIndex[j] + 1]++;
			A_transpose.val[index] = A.val[j];
			A_transpose.colIndex[index] = i;
		}
	}

	A_transpose.rowPtr.resize(A.m + 1);
}

void SLE_Solver_CRS_BICG(CRSMatrix &A, double *b, double eps, int max_iter, double *x, int &count) {
    int n = A.n;
    count = 0;

    double *r_curr = new double[n];
    double *r_next = new double[n];
    double *r_hat_curr = new double[n];
    double *r_hat_next = new double[n];

    double *p = new double[n];
    double *p_hat = new double[n];

    double alpha, beta;

    double r_norm;

    double *tmp_vec_1 = new double[n];
    double *tmp_vec_2 = new double[n];
    double dividend, divisor;

    CRSMatrix A_transpose;
    transpose_CRS_matrix(A, A_transpose);

#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        r_curr[i] = b[i];
        r_hat_curr[i] = r_curr[i];
        p[i] = r_curr[i];
        p_hat[i] = r_curr[i];
        x[i] = 0.0;
    }

    for (int iter = 0; iter < max_iter; ++iter) {
        ++count;
        
        mul_CRS_matrix_by_vector(A, p, tmp_vec_1);
		dividend = divisor = 0.0;
	#pragma parallel omp for schedule(static) reduction (+ : dividend, divisor)
		for (int i = 0; i < n; ++i) {
			dividend += r_curr[i] * r_hat_curr[i];
			divisor += tmp_vec_1[i] * p_hat[i];
		}
		alpha = dividend / divisor;

	#pragma omp parallel for schedule(static)
		for (int i = 0; i < n; ++i) {
			x[i] += alpha * p[i];
		}

		mul_CRS_matrix_by_vector(A_transpose, p_hat, tmp_vec_2);
	#pragma omp parallel for schedule(static)
		for (int i = 0; i < n; ++i) {
			r_next[i] = r_curr[i] - alpha * tmp_vec_1[i];
			r_hat_next[i] = r_hat_curr[i] - alpha * tmp_vec_2[i];
		}

        divisor = dividend;
		dividend = 0.0;
	#pragma parallel omp for schedule(static) reduction (+ : dividend)
		for (int i = 0; i < n; ++i) {
			dividend += r_next[i] * r_hat_next[i];
		}
		beta = dividend / divisor;

		r_norm = 0.0;
	#pragma parallel omp for schedule(static) reduction (+ : r_norm)
		for (int i = 0; i < n; ++i) {
			r_norm += r_next[i] * r_next[i];
		}
        r_norm = std::sqrt(r_norm);

		if (beta == 0.0 || r_norm < eps) {
			break;
		}

	#pragma omp parallel for schedule(static)
		for (int i = 0; i < n; ++i) {
			p[i] = r_next[i] + beta * p[i];
			p_hat[i] = r_hat_next[i] + beta * p_hat[i];
		}

		std::swap(r_curr, r_next);
		std::swap(r_hat_curr, r_hat_next);
    }

    delete [] r_curr;
    delete [] r_next;
    delete [] r_hat_curr;
    delete [] r_hat_next;

    delete [] p;
    delete [] p_hat;

    delete [] tmp_vec_1;
    delete [] tmp_vec_2;
}
