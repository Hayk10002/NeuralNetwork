#pragma once
#include <deque>
#include <functional>
#include <ctime>
#include <Windows.h>
#include <iomanip>
#include <conio.h>
#include "Matrix.hpp" //see my another repository named Matrices
double rand_double(double x);
double sigmoid(double x);
double sigmoid_deriv(double y);
double z(double x);
double z_deriv(double y);
void cls();
Matrix<double> multelembyelem(const Matrix<double> &a, const Matrix<double> &b);
Matrix<double> def_error_counting_alg(const Matrix<double> &a, const Matrix<double> &b);
class NNetwork
{
public:
	size_t layer_count;
	double ed_coeff;
	std::deque<Matrix<double>> layers, weights, errors;
	std::function<double(double)> activation_func = sigmoid, activation_func_deriv = sigmoid_deriv;
	std::function<Matrix<double>(const Matrix<double>&, const Matrix<double>&)> err_counting_alg;
	NNetwork(size_t lc, std::deque<size_t> ls, double ed_coeff = 0.3, std::function<double(double)> act_f = sigmoid, std::function<double(double)> act_f_d = sigmoid_deriv, std::function<Matrix<double>(const Matrix<double>&, const Matrix<double>& ) > err_c_a = def_error_counting_alg);

	void forward();
	void finderrors(const Matrix<double> &expected_res);
	void weightscorrection();
	void onecycle(const Matrix<double> &input_m, const Matrix<double> &output_res_m);
	double getallerrors();
	double getreserror();
	void printonscreen();
};

