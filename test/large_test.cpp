#include <iostream>
#include <cstdlib>

#include <BiCG.h>
#include <omp.h>

int main() {
    int n = 50000, nz = n;
    std::vector<double> val(n);
    std::vector<int> colIndex(n);
    std::vector<int> rowPtr(n + 1);
    double *b = new double[n];

    for (int i = 0; i < n; ++i) {
        b[i] = rand() % 10000;
        val[i] = rand() % 10000;
        colIndex[i] = rand() % n;
        rowPtr[i] = i;
    }
    rowPtr[n] = n;

    CRSMatrix A{ n, n, nz, val, colIndex, rowPtr };
    double *x = new double[n];
    double eps = 0.001;
    int max_iter = 2000, count;

    int num_threads[3] = { 1, 2, 4 };
    double start_time, end_time;
    double time[3];

    for (int i = 0; i < 3; ++i) {
        omp_set_num_threads(num_threads[i]);

        start_time = omp_get_wtime();
        SLE_Solver_CRS_BICG(A, b, eps, max_iter, x, count);
        end_time = omp_get_wtime();
        time[i] = end_time - start_time;

        std::cout << "Number of threads: " << num_threads[i]
                  << ", Time: " << time[i]
                  << ", Boost: " << time[0] / time[i] << "\n";

        std::cout << "Count iter = " << count << "\n";
    }

    delete [] b;
    delete [] x;

    return 0;
}
