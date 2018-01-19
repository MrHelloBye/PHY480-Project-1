#include <cstdio>

double** init_matrix(unsigned int);
void populate_matrix(double**);


int main()
{
    
    return 0;
}

double** init_matrix(unsigned int size)
{
    double** array = new double*[size];
    for (unsigned int i = 0; i<size; i++)
    {
        array[i] = new double[size];
        
        // Initialize elements to zero
        for (unsigned int j = 0; j<size; j++)
            array[i][j] = 0;
    }
    
    return array;
}