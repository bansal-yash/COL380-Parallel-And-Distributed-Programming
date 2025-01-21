#include <iostream>
#include <tuple>
using namespace std;

tuple<int, int, int, int, int, string, string> read_arguments(int argc, char *argv[])
{
    int type = stoi(argv[1]);
    int row1 = stoi(argv[2]);
    int col1 = stoi(argv[3]);
    int row2 = col1;
    int col2 = stoi(argv[4]);
    string input_path = argv[5];
    string output_path = argv[6];

    return {type, row1, col1, row2, col2, input_path, output_path};
}

double **createMatrix(int row, int col)
{
    double **mat = new double *[row];
    for (int i = 0; i < row; i++)
    {
        mat[i] = new double[col];
    }
    return mat;
}

void readMatrix(double **mat, int row, int col, string mat_path)
{
    double *inp_mtx = new double[row * col];
    FILE *fp = fopen(mat_path.c_str(), "rb");
    size_t temp = fread(inp_mtx, sizeof(double), row * col, fp);
    fclose(fp);

    int index = 0;
    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < col; j++)
        {
            mat[i][j] = inp_mtx[index];
            index++;
        }
    }

    delete[] inp_mtx;
}

void writeMatrix(double **mat, int row, int col, string mat_path)
{
    double *out_mtx = new double[row * col];

    int index = 0;
    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < col; j++)
        {
            out_mtx[index] = mat[i][j];
            index++;
        }
    }

    FILE *fp = fopen(mat_path.c_str(), "wb");
    fwrite(out_mtx, sizeof(double), row * col, fp);
    fclose(fp);

    delete[] out_mtx;
}

void deleteMatrix(double **mat, int row)
{
    for (int i = 0; i < row; i++)
    {
        delete[] mat[i];
    }

    delete[] mat;
}

// Matrix Multiplication functions
void matrixMultiplyIJK(double **mat_A, double **mat_B, double **mat_C, int row1, int col1, int col2)
{
    for (int i = 0; i < row1; i++)
    {
        for (int j = 0; j < col2; j++)
        {
            for (int k = 0; k < col1; k++)
            {
                mat_C[i][j] += mat_A[i][k] * mat_B[k][j];
            }
        }
    }
}

void matrixMultiplyIKJ(double **mat_A, double **mat_B, double **mat_C, int row1, int col1, int col2)
{
    for (int i = 0; i < row1; i++)
    {
        for (int k = 0; k < col1; k++)
        {
            for (int j = 0; j < col2; j++)
            {
                mat_C[i][j] += mat_A[i][k] * mat_B[k][j];
            }
        }
    }
}

void matrixMultiplyJIK(double **mat_A, double **mat_B, double **mat_C, int row1, int col1, int col2)
{
    for (int j = 0; j < col2; j++)
    {
        for (int i = 0; i < row1; i++)
        {
            for (int k = 0; k < col1; k++)
            {
                mat_C[i][j] += mat_A[i][k] * mat_B[k][j];
            }
        }
    }
}

void matrixMultiplyJKI(double **mat_A, double **mat_B, double **mat_C, int row1, int col1, int col2)
{
    for (int j = 0; j < col2; j++)
    {
        for (int k = 0; k < col1; k++)
        {
            for (int i = 0; i < row1; i++)
            {
                mat_C[i][j] += mat_A[i][k] * mat_B[k][j];
            }
        }
    }
}

void matrixMultiplyKIJ(double **mat_A, double **mat_B, double **mat_C, int row1, int col1, int col2)
{
    for (int k = 0; k < col1; k++)
    {
        for (int i = 0; i < row1; i++)
        {
            for (int j = 0; j < col2; j++)
            {
                mat_C[i][j] += mat_A[i][k] * mat_B[k][j];
            }
        }
    }
}

void matrixMultiplyKJI(double **mat_A, double **mat_B, double **mat_C, int row1, int col1, int col2)
{
    for (int k = 0; k < col1; k++)
    {
        for (int j = 0; j < col2; j++)
        {
            for (int i = 0; i < row1; i++)
            {
                mat_C[i][j] += mat_A[i][k] * mat_B[k][j];
            }
        }
    }
}

int main(int argc, char *argv[])
{
    // Reading command line arguments
    auto [type, row1, col1, row2, col2, input_path, output_path] = read_arguments(argc, argv);

    // Initialising three matrices mat_A, mat_B and mat_C as dynamic array of arrays
    double **mat_A = createMatrix(row1, col1);
    double **mat_B = createMatrix(row2, col2);
    double **mat_C = createMatrix(row1, col2);

    // Reading matrix A and B
    readMatrix(mat_A, row1, col1, input_path + "/mtx_A.bin");
    readMatrix(mat_B, row2, col2, input_path + "/mtx_B.bin");

    for (int i = 0; i < row1; i++)
    {
        for (int j = 0; j < col2; j++)
        {
            mat_C[i][j] = 0;
        }
    }

    // Matrix multiplication code call
    if (type == 0)
    {
        matrixMultiplyIJK(mat_A, mat_B, mat_C, row1, col1, col2);
    }
    if (type == 1)
    {
        matrixMultiplyIKJ(mat_A, mat_B, mat_C, row1, col1, col2);
    }
    if (type == 2)
    {
        matrixMultiplyJIK(mat_A, mat_B, mat_C, row1, col1, col2);
    }
    if (type == 3)
    {
        matrixMultiplyJKI(mat_A, mat_B, mat_C, row1, col1, col2);
    }
    if (type == 4)
    {
        matrixMultiplyKIJ(mat_A, mat_B, mat_C, row1, col1, col2);
    }
    if (type == 5)
    {
        matrixMultiplyKJI(mat_A, mat_B, mat_C, row1, col1, col2);
    }

    // Writing the resulting matrix
    writeMatrix(mat_C, row1, col2, output_path + "/mtx_C.bin");

    // Deleting the matrices
    deleteMatrix(mat_A, row1);
    deleteMatrix(mat_B, row2);
    deleteMatrix(mat_C, row1);
}