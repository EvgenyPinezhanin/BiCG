#include <iostream>
#include <random>

#include <BiCG.h>
#include <omp.h>

void print_vector(double *x, int n) {
    for (int i = 0; i < n; ++i) {
        std::cout << x[i] << " ";
    }
    std::cout << "\n";
}

int main() {
    int n = 12, nz = 17;
    std::vector<double> val{ 1.0, 2.0, 1.0, 3.0, 4.0, 2.0, 5.0, 6.0, 2.0, 7.0, 3.0, 8.0, 9.0, 3.0, 10.0, 11.0, 12.0 };
    std::vector<int> colIndex{ 0, 1, 8, 2, 3, 10, 4, 5, 3, 6, 10, 7, 8, 3, 9, 10, 11 };
    std::vector<int> rowPtr{ 0, 1, 3, 4, 6, 7, 8, 11, 12, 13, 15, 16, 17 };
    CRSMatrix A{ n, n, nz, val, colIndex, rowPtr };
    double b[n] = { 3, 4, 5, 8, 1, 7, 2, 7, 9, 1, 3, 5 }, x[n], x_copy[n];
    double eps = 0.01;
    int max_iter = 15, count;

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

        std::cout << "x = ";
        print_vector(x, n);
        std::cout << "Count iter = " << count << "\n";
    }

    return 0;
}
