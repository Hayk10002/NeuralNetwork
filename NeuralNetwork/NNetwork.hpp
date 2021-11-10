#pragma once
#include <deque>
#include <functional>
#include <ctime>
#include <Windows.h>
#include <iomanip>
#include <conio.h>
#include <fstream>
#include <thread>
#include "Matrix.hpp" //see my another repository named Matrices
struct Act_func { std::function<double(double)> f, f_deriv; };
double rand_double(double x);
double sigmoid_f(double x);
double sigmoid_f_deriv(double y);
double z_f(double x);
double z_f_deriv(double y);
extern Act_func sigmoid, z;
void cls();
Matrix<double> multelembyelem(const Matrix<double> &a, const Matrix<double> &b);
Matrix<double> def_error_counting_alg(const Matrix<double> &a, const Matrix<double> &b);
class NNetwork
{
	size_t layer_count;
	double l_rate;
	std::deque<Matrix<double>> layers, weights, errors;
	Act_func activ_func = sigmoid;
	std::function<Matrix<double>(const Matrix<double>&, const Matrix<double>&)> err_counting_alg = def_error_counting_alg;
	std::deque<Matrix<double>> return_dweights();

public:
	NNetwork(const NNetwork& other);
	NNetwork(size_t lc, std::deque<size_t> ls, double l_rate = 0.3);
	void set_activ_func(Act_func act_f);
	void set_error_counting_algorithm(std::function<Matrix<double>(const Matrix<double>&, const Matrix<double>&)> err_c_a);
	void set_learning_rate(double l_r);
	void forward(const Matrix<double> &input);
	void find_errors(const Matrix<double> &expected_res);
	void weights_correction();
	void one_learning_cycle(const Matrix<double> &input_m, const Matrix<double> &output_res_m);
	void learning_cycles_threaded(size_t cycles_count, const std::deque<Matrix<double>>&  input_ms, const std::deque<Matrix<double>>&  output_res_ms);
	double get_all_errors();
	double get_final_error();
	Matrix<double> pass_input(const Matrix<double> &input);
	void print_on_screen();
	//friend class NNetwork;
	friend std::ifstream& operator>>(std::ifstream&, NNetwork&);
	friend std::ofstream& operator<<(std::ofstream&, const NNetwork&);

};


