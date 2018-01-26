#include <cstdio>
#include "xtensor/xtensor.hpp"
#include "xtensor/xarray.hpp"
#include "xtensor/xmath.hpp"
#include "xtensor/xio.hpp"

void populate_matrix(double**, int);


int main()
{
    //Initialize tridiagonal symmetrix matrix equation
    unsigned int dim = 5;
    
    xt::xarray<double> diag = xt::ones<double>({dim});
    xt::xarray<double> upper_diag = xt::ones<double>({dim-1});
    
    xt::xarray<double> condition_vector = xt::zeros<double>({dim});
    xt::xarray<double> solution_vector = xt::zeros<double>({dim});
    
    // Solve matrix equation
    xt::xarray<double> *thing = new xt::xarray<double>;
    
    return 0;
}

void populate_matrix(double** array, int dim)
{
    for (int i = 0; i<dim; i++)
    {
        for (int j = 0; j<dim; j++)
        {
            if (i==j)
                array[i][j] = 2;
            else if (i==j+1 || i==j-1)
                array[i][j] = -1;
        }
    }
}