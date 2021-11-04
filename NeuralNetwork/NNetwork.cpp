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
Matrix<double> multelembyelem(Matrix<double> a, Matrix<double> b)
{
	Matrix<double> res = a;
	b.resize(a.get_rows_count(), a.get_columns_count(), 1);
	for (size_t i = 0; i < a.get_rows_count(); i++) for (size_t j = 0; j < a.get_columns_count(); j++) res(i, j) *= b(i, j);
	return res;

}
NNetwork::NNetwork(size_t lc, std::deque<size_t> ls, double ed_coeff) : layer_count(lc), ed_coeff(ed_coeff)
{
	srand((size_t)time(0));

	layers.resize(lc, Matrix<double>(1, 1));
	for (size_t i = 0; i < lc; i++) layers[i] = Matrix<double>(1, ls[i]);

	errors.resize(lc - 1, Matrix<double>(1, 1));
	for (size_t i = 0; i < lc - 1; i++) errors[i] = Matrix<double>(1, ls[i + 1]);

	weights.resize(lc - 1, Matrix<double>(1, 1));
	for (size_t i = 0; i < lc - 1; i++)
	{
		weights[i] = Matrix<double>(ls[i], ls[i + 1]);
		weights[i].apply_func(rand_double);
	}


}

void NNetwork::forward()
{
	for (size_t i = 1; i < layer_count; i++)
	{
		Matrix<double> layer(layers[i - 1] * weights[i - 1]);
		layer.apply_func(activation_func);
		layers[i] = layer;
	}
}

void NNetwork::finderrors(Matrix<double> expected_res)
{
	errors[layer_count - 2] = -(layers[layer_count - 1] - expected_res);
	for (size_t i = layer_count - 3; (int)i >= 0; i--)
	{
		errors[i] = errors[i + 1] * weights[i + 1].get_Transpose();
	}
}

void NNetwork::weightscorrection()
{
	for (size_t i = 0; i < layer_count - 1; i++)
	{
		Matrix<double> layer = layers[i + 1];
		layer.apply_func(activation_func_deriv);
		Matrix<double> dweight(layers[i].get_Transpose() * (ed_coeff * multelembyelem(layer, errors[i])));
		weights[i] += dweight;
	}
}

void NNetwork::onecycle(Matrix<double> input_m, Matrix<double> output_res_m)
{
	input_m.resize(layers[0].get_rows_count(), layers[0].get_columns_count());
	layers[0] = input_m;
	forward();
	finderrors(output_res_m);
	weightscorrection();

}

double NNetwork::geterror()
{
	double err = 0;
	for (size_t i = 0; i < errors[layer_count - 2].get_columns_count(); i++) err += errors[layer_count - 2](0, i) * errors[layer_count - 2](0, i);
	return err;
}

void NNetwork::printonscreen()
{
	system("cls");
	for (size_t i = 0; i < layer_count; i++)
	{
		std::cout << "Value:\n" << layers[i] << "\n\n";
		if (i > 0) std::cout << "Error:\n" << errors[i - 1] << "\n\n";
		if (i < layer_count - 1) std::cout << "Weights:\n" << weights[i] << "\n\n\n\n";
	}
	std::cout << "Error: " << geterror();
}
