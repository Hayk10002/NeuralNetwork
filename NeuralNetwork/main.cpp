#include <iostream>
#include <conio.h>
#include "NNetwork.hpp"
#include "Matrix.hpp"
using namespace std;
Matrix<double> database[16][2] = 
{
	{Matrix<double>(1, 9, {{1, 1, 1, 0, 0, 0, 0, 0, 0}}), Matrix<double>(1, 2, {{0.8, 0.2}})},
	{Matrix<double>(1, 9, {{0, 0, 0, 1, 1, 1, 0, 0, 0}}), Matrix<double>(1, 2, {{0.8, 0.2}})},
	{Matrix<double>(1, 9, {{0, 0, 0, 0, 0, 0, 1, 1, 1}}), Matrix<double>(1, 2, {{0.8, 0.2}})},
	{Matrix<double>(1, 9, {{1, 0, 0, 1, 0, 0, 1, 0, 0}}), Matrix<double>(1, 2, {{0.2, 0.8}})},
	{Matrix<double>(1, 9, {{0, 1, 0, 0, 1, 0, 0, 1, 0}}), Matrix<double>(1, 2, {{0.2, 0.8}})},
	{Matrix<double>(1, 9, {{0, 0, 1, 0, 0, 1, 0, 0, 1}}), Matrix<double>(1, 2, {{0.2, 0.8}})},
	{Matrix<double>(1, 9, {{1, 1, 1, 1, 0, 0, 1, 0, 0}}), Matrix<double>(1, 2, {{0.8, 0.8}})},
	{Matrix<double>(1, 9, {{1, 1, 1, 0, 1, 0, 0, 1, 0}}), Matrix<double>(1, 2, {{0.8, 0.8}})},
	{Matrix<double>(1, 9, {{1, 1, 1, 0, 0, 1, 0, 0, 1}}), Matrix<double>(1, 2, {{0.8, 0.8}})},
	{Matrix<double>(1, 9, {{1, 0, 0, 1, 1, 1, 1, 0, 0}}), Matrix<double>(1, 2, {{0.8, 0.8}})},
	{Matrix<double>(1, 9, {{0, 1, 0, 1, 1, 1, 0, 1, 0}}), Matrix<double>(1, 2, {{0.8, 0.8}})},
	{Matrix<double>(1, 9, {{0, 0, 1, 1, 1, 1, 0, 0, 1}}), Matrix<double>(1, 2, {{0.8, 0.8}})},
	{Matrix<double>(1, 9, {{1, 0, 0, 1, 0, 0, 1, 1, 1}}), Matrix<double>(1, 2, {{0.8, 0.8}})},
	{Matrix<double>(1, 9, {{0, 1, 0, 0, 1, 0, 1, 1, 1}}), Matrix<double>(1, 2, {{0.8, 0.8}})},
	{Matrix<double>(1, 9, {{0, 0, 1, 0, 0, 1, 1, 1, 1}}), Matrix<double>(1, 2, {{0.8, 0.8}})},
	{Matrix<double>(1, 9, {{0, 0, 0, 0, 0, 0, 0, 0, 0}}), Matrix<double>(1, 2, {{0.2, 0.2}})}
};
Matrix<double> err_c_a(const Matrix<double> &res, const Matrix<double> &exp_res)
{
	Matrix<double> exp_r = exp_res;
	size_t r = res.get_rows_count(), c = res.get_columns_count();
	Matrix<double> err(r, c);
	exp_r.resize(r, c);
	for (size_t i = 0; i < r; i++) for (size_t j = 0; j < c; j++)
	{
		if (exp_res(i, j) > 0.5) err(i, j) = ((res(i, j) > exp_res(i, j)) ? 0.0 : (exp_res(i, j) - res(i, j)));
		else if(exp_res(i, j) < 0.5) err(i, j) = ((res(i, j) < exp_res(i, j)) ? 0.0 : (exp_res(i, j) - res(i, j)));
	}
	return err;
}
int main()
{
	NNetwork nn(4, {9, 5, 4, 2}, 0.5, sigmoid, sigmoid_deriv, err_c_a);
	double prse = 0.0, prre = 0.0;
	for (int step = 0; ;step++)
	{
		double sum_err = 0.0, res_err = 0.0;
		for (int i = 0; i < 16; i++)
		{
			nn.onecycle(database[i][0], database[i][1]);
			sum_err += nn.getallerrors();
			res_err += nn.getreserror();
			
		}
		nn.printonscreen();
		cout << "\n\n";
		cout << prse << '\t' << prre << '\t' << step << endl;
		prse = sum_err; 
		prre = res_err;
		

	}

	return 0;
}