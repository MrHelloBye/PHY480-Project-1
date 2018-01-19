#include <cstdio>

double** init_matrix(unsigned int);
void populate_matrix(double**);


int main()
{
    
    return 0;
}

double** init_matrix(unsigned int dim)
{
    double** array = new double*[dim];
    for (unsigned int i = 0; i<dim; i++)
    {
        array[i] = new double[dim];
        
        // Initialize elements to zero
        for (unsigned int j = 0; j<dim; j++)
            array[i][j] = 0;
    }
    
    return array;
}

