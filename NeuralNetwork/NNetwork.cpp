#include "NNetwork.hpp"


double rand_double(double x)
{
	return (double)rand() / (double)RAND_MAX;
}
double sigmoid(double x)
{
	return 1.0 / (1.0 + exp(-x));
}
double sigmoid_deriv(double y) // in terms of sigmoid(x)
{
	return y * (1.0 - y);
}
double z(double x)
{
	if (x < 0) return x / 100.0;
	if (x < 1) return x;
	return 1 + (x - 1) / 100.0;
}
double z_deriv(double y)
{
	if (y < 0 || y > 1) return 0.1;
	return 1.0;
}
void cls()
{
	HANDLE console = ::GetStdHandle(STD_OUTPUT_HANDLE);
	CONSOLE_SCREEN_BUFFER_INFO csbi;
	::GetConsoleScreenBufferInfo(console, &csbi);
	COORD origin = { 0, 0 };
	/*DWORD written;
	::FillConsoleOutputCharacterA(console, ' ', csbi.dwSize.X * csbi.dwSize.Y,
		origin, &written);
	::FillConsoleOutputAttribute(console, csbi.wAttributes, csbi.dwSize.X * csbi.dwSize.Y,
		origin, &written);*/
	::SetConsoleCursorPosition(console, origin);
}
Matrix<double> multelembyelem(const Matrix<double> &a, const Matrix<double> &b)
{
	Matrix<double> res(a.get_rows_count(), a.get_columns_count()), b1 = b;
	b1.resize(a.get_rows_count(), a.get_columns_count(), 1);
	for (size_t i = 0; i < a.get_rows_count(); i++) for (size_t j = 0; j < a.get_columns_count(); j++) res(i, j) = a(i, j) * b1(i, j);
	return res;

}
Matrix<double> def_error_counting_alg(const Matrix<double>& a, const Matrix<double>& b)
{
	return -(a - b);
}
NNetwork::NNetwork(size_t lc, std::deque<size_t> ls, double ed_coeff, std::function<double(double)> act_f, std::function<double(double)> act_f_d, std::function<Matrix<double>(const Matrix<double>&, const Matrix<double>&) > err_c_a) :
	layer_count(lc),
	ed_coeff(ed_coeff),
	activation_func(act_f),
	activation_func_deriv(act_f_d),
	err_counting_alg(err_c_a)
{
	srand((size_t)time(0));

	layers.resize(lc, Matrix<double>(1, 1));
	for (size_t i = 0; i < lc; i++) layers[i] = Matrix<double>(1, ls[i] + (i != lc - 1), 1);

	errors.resize(lc - 1, Matrix<double>(1, 1));
	for (size_t i = 0; i < lc - 1; i++) errors[i] = Matrix<double>(1, ls[i + 1]);

	weights.resize(lc - 1, Matrix<double>(1, 1));
	for (size_t i = 0; i < lc - 1; i++)
	{
		weights[i] = Matrix<double>(ls[i] + 1, ls[i + 1] );
		weights[i].apply_func(rand_double);
	}


}

void NNetwork::forward()
{
	for (size_t i = 1; i < layer_count; i++)
	{
		Matrix<double> layer(layers[i - 1] * weights[i - 1]);
		layer.apply_func(activation_func);
		layer.resize(layers[i].get_rows_count(), layers[i].get_columns_count(), 1);
		layers[i] = layer;
	}
}

void NNetwork::finderrors(const Matrix<double> &expected_res)
{
	errors[layer_count - 2] = err_counting_alg(layers[layer_count - 1], expected_res);
	for (size_t i = layer_count - 3; (int)i >= 0; i--)
	{
		Matrix<double> weight = weights[i + 1].delete_Row(weights[i + 1].get_rows_count() - 1);
		errors[i] = errors[i + 1] * weight.get_Transpose();
	}
}

void NNetwork::weightscorrection()
{
	for (size_t i = 0; i < layer_count - 1; i++)
	{
		Matrix<double> layer = layers[i + 1];
		layer.resize(layer.get_rows_count(), layer.get_columns_count() - (i != layer_count - 2));
		layer.apply_func(activation_func_deriv);
		Matrix<double> dweight(layers[i].get_Transpose() * (ed_coeff * multelembyelem(layer, errors[i])));
		weights[i] += dweight;
	}
}

void NNetwork::onecycle(const Matrix<double> &input_m, const Matrix<double> &output_res_m)
{
	Matrix<double> inpm = input_m;
	inpm.resize(layers[0].get_rows_count(), layers[0].get_columns_count(), 1);
	layers[0] = inpm;
	forward();
	finderrors(output_res_m);
	weightscorrection();

}

double NNetwork::getallerrors()
{
	double err = 0;
	for (size_t i = 0; i < layer_count - 1; i++) for (size_t j = 0; j < errors[i].get_columns_count(); j++) err += errors[i](0, j) * errors[i](0, j);
	return err;
}

double NNetwork::getreserror()
{
	double err = 0;
	for (size_t i = 0; i < errors[layer_count - 2].get_columns_count(); i++) err += errors[layer_count - 2](0, i) * errors[layer_count - 2](0, i);
	return err;
}

void NNetwork::printonscreen()
{
	cls();
	std::cout.precision(4);
	std::cout.setf(std::ios::fixed);
	for (size_t i = 0; i < layer_count; i++)
	{
		std::cout << "Value:\n";layers[i].print("\t"); std::cout << "\n";
		if (i > 0) { std::cout << "Error:\n"; errors[i - 1].print("\t"); std::cout << "\n"; }
		if (i < layer_count - 1) { std::cout << "Weights:\n"; weights[i].print("\t"); std::cout << "\n\n\n\n"; }
	}
	std::cout << "Error: " << getallerrors();
}
