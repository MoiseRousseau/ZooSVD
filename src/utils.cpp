#include <iostream>
#include <cmath>

extern "C" {
    void print_matrixd(double*, long int[2], bool);
    void print_matrixf(float*, long int[2], bool);    
    double matrix_norm(double*, long int[2]);
}

void print_matrixd(double* mat_val, long int shape[2], bool is_complex)
{
    size_t row_length = shape[1];
    if (is_complex) row_length *= 2;
    for (size_t i=0; i<(size_t)shape[0]; i++)
    {
        for (size_t j=0; j<row_length; j++) {
            std::cout << *(mat_val + row_length*i+j);
            if (is_complex) {
                j++;
                std::cout << "+" << *(mat_val + row_length*i+j) << "j";
            }
            std::cout << " ";
        }
        std::cout << std::endl;
    }
    return;
}

void print_matrixf(float* mat_val, long int shape[2], bool is_complex)
{
    size_t row_length = shape[1];
    if (is_complex) row_length *= 2;
    for (size_t i=0; i<(size_t)shape[0]; i++)
    {
        for (size_t j=0; j<row_length; j++) {
            std::cout << *(mat_val + row_length*i+j);
            if (is_complex) {
                j++;
                std::cout << "+" << *(mat_val + row_length*i+j) << "j";
            }
            std::cout << " ";
        }
        std::cout << std::endl;
    }
    return;
}

//template <typename T>
double matrix_norm(double* mat_val, long int shape[2])
{
    double sum = 0;
    double val;
    for (size_t i=0; i<(size_t)shape[0]; i++)
    {
        for (size_t j=0; j<(size_t)shape[1]; j++) {
            val = *(mat_val + shape[1]*i+j);
            sum += val*val;
        }
    }
    return sqrt(sum);
}
