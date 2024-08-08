#ifndef _BiCG_H_
#define _BiCG_H_

#include <vector>

struct CRSMatrix
{
    int n; // Число строк в матрице
    int m; // Число столбцов в матрице
    int nz; // Число ненулевых элементов в разреженной матрице
    std::vector<double> val; // Массив значений матрицы по строкам
    std::vector<int> colIndex; // Массив номеров столбцов
    std::vector<int> rowPtr; // Массив индексов начала строк
};

void SLE_Solver_CRS_BICG(CRSMatrix & A, double *b, double eps, int max_iter, double *x, int &count);

#endif // _BiCG_H_
