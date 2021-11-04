#pragma once
#include <deque>
#include <functional>
#include <ctime>
#include "Matrix.hpp" //see my another repository named Matrices
double rand_double(double x);
double sigmoid(double x);
double sigmoid_deriv(double x);
Matrix<double> multelembyelem(Matrix<double> a, Matrix<double> b);
class NNetwork
{
public:
	size_t layer_count;
	double ed_coeff;
	std::deque<Matrix<double>> layers, weights, errors;
	std::function<double(double)> activation_func = sigmoid, activation_func_deriv = sigmoid_deriv;
	NNetwork(size_t lc, std::deque<size_t> ls, double ed_coeff = 0.3);

	void forward();
	void finderrors(Matrix<double> expected_res);
	void weightscorrection();
	void onecycle(Matrix<double> input_m, Matrix<double> output_res_m);
	double geterror();
	void printonscreen();
};

